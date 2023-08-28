import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, WhisperFeatureExtractor, XLMRobertaTokenizer
try:
    import wandb
except ImportError:
    wandb = None

from args import parse_args_main
from datasets import SymileDataset
from models import AudioEncoder, ImageEncoder, TextEncoder, SymileModel
from src.losses import pairwise_infonce, symile
from utils import seed_all, wandb_init


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
                - audio_attention_masks: torch.Tensor of shape (batch_sz, 3000)
                - image_pixel_values: torch.Tensor of shape (batch_sz, 3, 224, 224)
                - text_input_ids: torch.Tensor of shape (batch_sz, len_longest_seq)
                - text_attention_masks: torch.Tensor of shape (batch_sz, len_longest_seq)
                - templates: list of length batch_sz containing the template number
        """
        audio_input_features = torch.stack([s["audio"]["input_features"] for s in batch])
        audio_attention_masks = torch.stack([s["audio"]["attention_mask"] for s in batch])

        image_pixel_values = torch.stack([s["image"]["pixel_values"] for s in batch])

        text_list = [s["text"] for s in batch]
        text = self.txt_tokenizer(text=text_list, return_tensors="pt",
                                  padding=True, truncation=True)

        templates = [s["template"] for s in batch]

        return {"audio_input_features": audio_input_features,
                "audio_attention_masks": audio_attention_masks,
                "image_pixel_values": image_pixel_values,
                "text_input_ids": text["input_ids"],
                "text_attention_masks": text["attention_mask"],
                "templates": templates}


def load_data(args):
    audio_feat_extractor = WhisperFeatureExtractor.from_pretrained(args.audio_model_id)
    img_processor = CLIPImageProcessor.from_pretrained(args.image_model_id)
    txt_tokenizer = XLMRobertaTokenizer.from_pretrained(args.text_model_id)

    df = pd.read_csv(args.dataset_path)
    df["text"] = df.text.fillna("")
    ds = SymileDataset(df, audio_feat_extractor, img_processor)
    return DataLoader(ds, batch_size=args.batch_sz, shuffle=True, collate_fn=Collator(txt_tokenizer))


def pretrain(args, symile_model):
    dl = load_data(args)
    loss_fn = symile if args.loss_fn == "symile" else pairwise_infonce
    optimizer = torch.optim.AdamW(symile_model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for data in dl:
            breakpoint()
            audio_output = audio_encoder(input_features=data["audio_input_features"], attention_mask=data["audio_attention_masks"])
            image_output = image_encoder(pixel_values=data["image_pixel_values"])
            text_output = text_encoder(input_ids=data["text_input_ids"], attention_mask=data["text_attention_masks"])

            # r_a, r_i, r_t, logit_scale_exp = symile_model(**data)
            # audio = audio_encoder(audio)
            # image = image_encoder(**data["image"])
            # text = text_encoder(**data["text"])

            # loss = loss_fn(r_a, r_b, r_c, logit_scale_exp, args.normalize)

if __name__ == '__main__':
    # TODO:
    # - maybe move all data-related functions into a data_utils.py file?

    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'
    args = parse_args_main()
    wandb_init(args)
    if args.use_seed:
        seed_all(args.seed)

    # PRETRAIN
    print("\n\n\n...pretraining...\n")
    audio_encoder = AudioEncoder(args.audio_model_id)
    image_encoder = ImageEncoder(args.image_model_id)
    text_encoder = TextEncoder(args.text_model_id)
    symile_model = SymileModel(audio_encoder, image_encoder, text_encoder,
                               args.logit_scale_init)
    pretrain(args, symile_model)
    # TODO: make sure that at the end of this, all the encoders have actually
    # been trained (weights changed)

    # EVALUATE