import os

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, CLIPImageProcessor, \
                         WhisperFeatureExtractor, XLMRobertaTokenizer
try:
    import wandb
except ImportError:
    wandb = None

from args import parse_args_main
from datasets import SymileDataset
from models import AudioEncoder, ImageEncoder, TextEncoder, SymileModel
from src.losses import pairwise_infonce, symile
from utils import seed_all, wandb_init


def print_gpu_info():
    for i in range(torch.cuda.device_count()):
        t = round(torch.cuda.get_device_properties(i).total_memory/1024**3, 3)
        r = round(torch.cuda.memory_reserved(i)/1024**3, 3)
        a = round(torch.cuda.memory_allocated(i)/1024**3, 3)
        print("\nGPU: ", i)
        print("\n   total_memory: ", t)
        print("\n   memory_reserved: ", r)
        print("\n   memory_allocated: ", a)
    return


def print_num_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    num_params_mill = '{:.2f}'.format(num_params/1000000)
    print(num_params_mill, "M parameters")
    return


class Collator:
    """
    Custom collate function so that the text tokenizer can be called on a batch
    of text, which is then padded to the length of the longest sequence in the batch.
    """
    def __init__(self, txt_tokenizer):
        self.txt_tokenizer = txt_tokenizer
    def __call__(self, batch):
        """
        Args:
            batch (list): List of data samples of length `batch_sz`. Each sample
                          is a dictionary with keys `audio`, `image`, `text`,
                          and `template` (see SymileDataset.__getitem__).
        Returns:
            (dict): of batched data samples with the following keys:
                - audio_input_features: torch.Tensor of shape (batch_sz, 80, 3000)
                - audio_attention_mask: torch.Tensor of shape (batch_sz, 3000)
                - image_pixel_values: torch.Tensor of shape (batch_sz, 3, 224, 224)
                - text_input_ids: torch.Tensor of shape (batch_sz, len_longest_seq)
                - text_attention_mask: torch.Tensor of shape (batch_sz, len_longest_seq)
                - templates: torch.Tensor of shape (batch_sz) containing template numbers
        """
        audio_input_features = torch.stack([s["audio"]["input_features"] for s in batch])
        audio_attention_mask = torch.stack([s["audio"]["attention_mask"] for s in batch])

        image_pixel_values = torch.stack([s["image"]["pixel_values"] for s in batch])

        text_list = [s["text"] for s in batch]
        text = self.txt_tokenizer(text=text_list, return_tensors="pt",
                                  padding=True, truncation=True)

        templates = torch.Tensor([s["template"] for s in batch])

        return {"audio_input_features": audio_input_features,
                "audio_attention_mask": audio_attention_mask,
                "image_pixel_values": image_pixel_values,
                "text_input_ids": text["input_ids"],
                "text_attention_mask": text["attention_mask"],
                "templates": templates}


def load_data(args):
    """
    Note that encoder features are taken from the EOS or BOS embedding.

    transformers' CLIP and ImageBind take features from EOS embedding:
    - https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/models/clip/modeling_clip.py#L757C4-L757C4
    - https://github.com/facebookresearch/ImageBind/blob/95d27c7fd5a8362f3527e176c3a80ae5a4d880c0/imagebind/models/imagebind_model.py#L384C9-L384C9

    mBERT and XLM-Roberta take features from BOS embedding:
    - https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/models/bert/modeling_bert.py#L661
    - https://github.com/huggingface/transformers/blob/41aef33758ae166291d72bc381477f2db84159cf/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L580

       for BERT and XLM-Roberta,
    """
    audio_feat_extractor = WhisperFeatureExtractor.from_pretrained(args.audio_model_id)
    img_processor = CLIPImageProcessor.from_pretrained(args.image_model_id)

    if args.text_model_id == "bert-base-multilingual-cased":
        txt_tokenizer = BertTokenizer.from_pretrained(args.text_model_id)
        if args.text_embedding == "eos":
            args.feat_token_id = txt_tokenizer.sep_token_id
        elif args.text_embedding == "bos":
            args.feat_token_id = txt_tokenizer.cls_token_id
    elif args.text_model_id == "xlm-roberta-base":
        txt_tokenizer = XLMRobertaTokenizer.from_pretrained(args.text_model_id)
        if args.text_embedding == "eos":
            args.feat_token_id = txt_tokenizer.eos_token_id
        elif args.text_embedding == "bos":
            args.feat_token_id = txt_tokenizer.bos_token_id

    df = pd.read_csv(args.dataset_path)
    df["text"] = df.text.fillna("")
    ds = SymileDataset(df, audio_feat_extractor, img_processor)
    num_workers = len(os.sched_getaffinity(0)) # from max_num_worker_suggest in DataLoader docs
    return DataLoader(ds, batch_size=args.batch_sz, shuffle=True,
                      num_workers=num_workers, collate_fn=Collator(txt_tokenizer))


def pretrain(args, model, dl):
    loss_fn = symile if args.loss_fn == "symile" else pairwise_infonce
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        if epoch % 1 == 0:
            print("    epoch: ", epoch, "\n")

        for data in dl:
            data = {k: v.to(args.device) for k, v in data.items()}
            r_a, r_i, r_t, logit_scale_exp = model(data)

            loss = loss_fn(r_a, r_i, r_t, logit_scale_exp,
                           args.normalize, args.device)
            loss.backward()
            optimizer.step()
            if args.wandb:
                wandb.log({"pretrain_loss": loss,
                           "logit_scale_exp": logit_scale_exp})
            optimizer.zero_grad()

if __name__ == '__main__':
    # TODO:
    # - maybe move all data-related functions into a data_utils.py file?
    # - maybe move all eval functions into a evals file?
    # - move print functions into a utils.py file?

    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'
    args = parse_args_main()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb_init(args)
    if args.use_seed:
        seed_all(args.seed)

    # PRETRAIN
    print("\n\n...pretraining...\n")
    dl = load_data(args)
    # TODO: do you need these encoders initialized out here?
    audio_encoder = AudioEncoder(args.audio_model_id, args.d)
    image_encoder = ImageEncoder(args.image_model_id, args.d)
    text_encoder = TextEncoder(args.text_model_id, args.d, args.feat_token_id)
    symile_model = SymileModel(audio_encoder, image_encoder, text_encoder,
                               args.logit_scale_init)
    if torch.cuda.device_count() > 1:
        print("using", torch.cuda.device_count(), "GPUs\n")
        symile_model = nn.DataParallel(symile_model)
    symile_model.to(args.device)

    # for model in [audio_encoder, image_encoder, text_encoder]:
        # print_num_params(model)

    pretrain(args, symile_model, dl)

    # EVALUATE
    if args.evaluation == "zeroshot_clf":
        print("\n\n...evaluation: zero-shot classification...\n")
        test_zeroshot_clf(args, symile_model)
    # elif args.evaluation == "support_clf":
    #     print("\n\n\n...evaluation: in support classification...\n")
    #     test_support_clf(args, encoders)