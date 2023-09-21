import os
from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, CLIPImageProcessor, \
                         WhisperFeatureExtractor, XLMRobertaTokenizer
from transformers import BertModel, CLIPVisionModel, WhisperModel, XLMRobertaModel

from args import parse_args_pretrain
from datasets import SymileDataset


def split_train(split):
    read_dir = Path("/gpfs/scratch/as16583/tensors")
    audio = torch.load(read_dir / f"audio_{split}.pt")
    image = torch.load(read_dir / f"image_{split}.pt")
    text = torch.load(read_dir / f"text_{split}.pt")

    permutation = torch.randperm(len(audio))
    audio = audio[permutation]
    image = image[permutation]
    text = text[permutation]

    save_dir = Path("/gpfs/scratch/as16583/random_tensors")

    audio_train = audio[0:30000]
    audio_val = audio[30000:]
    torch.save(audio_train, save_dir / f"audio_train.pt")
    torch.save(audio_val, save_dir / f"audio_val.pt")

    image_train = image[0:30000]
    image_val = image[30000:]
    torch.save(image_train, save_dir / f"image_train.pt")
    torch.save(image_val, save_dir / f"image_val.pt")

    text_train = text[0:30000]
    text_val = text[30000:]
    torch.save(text_train, save_dir / f"text_train.pt")
    torch.save(text_val, save_dir / f"text_val.pt")


class Collator:
    """
    Custom collate function so that the text tokenizer can be called on a batch
    of text, which is then padded to the length of the longest sequence in the batch.
    """
    def __init__(self, txt_tokenizer, max_length=512):
        self.max_length = max_length
        self.txt_tokenizer = txt_tokenizer
    def __call__(self, batch):
        """
        Args:
            batch (list): List of data samples of length `batch_sz`. Each sample
                          is a dictionary with keys `audio`, `image`, `text`,
                          `template`, `idx`, and (optionally) `in_support`
                          (see SymileDataset.__getitem__).
        Returns:
            (dict): of batched data samples with the following keys:
                - audio_input_features: torch.Tensor of shape (batch_sz, 80, 3000)
                - audio_attention_mask: torch.Tensor of shape (batch_sz, 3000)
                - image_pixel_values: torch.Tensor of shape (batch_sz, 3, 224, 224)
                - text_input_ids: torch.Tensor of shape (batch_sz, len_longest_seq)
                - text_attention_mask: torch.Tensor of shape (batch_sz, len_longest_seq)
                - templates: torch.Tensor of shape (batch_sz) containing template numbers
                - idx: torch.Tensor of shape (batch_sz) with unique identifier for data sample
                - in_support: (optional) torch.Tensor of shape (batch_sz) where 1 means
                              sample is in support, 0 otherwise.
        """
        audio_input_features = torch.stack([s["audio"]["input_features"] for s in batch])
        audio_attention_mask = torch.stack([s["audio"]["attention_mask"] for s in batch])

        image_pixel_values = torch.stack([s["image"]["pixel_values"] for s in batch])

        text_list = [s["text"] for s in batch]
        text = self.txt_tokenizer(text=text_list, return_tensors="pt",
                                  padding="max_length", max_length=self.max_length,
                                  truncation=True)
                                #   padding=True, truncation=True)

        templates = torch.Tensor([s["template"] for s in batch])
        idx = torch.Tensor([s["idx"] for s in batch])

        batched_data = {"audio_input_features": audio_input_features,
                        "audio_attention_mask": audio_attention_mask,
                        "image_pixel_values": image_pixel_values,
                        "text_input_ids": text["input_ids"],
                        "text_attention_mask": text["attention_mask"],
                        "templates": templates,
                        "idx": idx}

        if "in_support" in batch[0]:
            batched_data["in_support"] = torch.Tensor([s["in_support"] for s in batch])

        return batched_data


def get_full_data_paths(df, args):
    def _full_audio_path(r):
        if r.template in [1]:
            return args.data_dir_commonvoice / r.audio_path.strip("/")
        else:
            return args.data_dir_generated_audio / r.audio_path.strip("/")
    df["audio_path"] = df.apply(lambda r: _full_audio_path(r), axis=1)

    def _full_image_path(r):
        if r.template in [1, 3]:
            return args.data_dir_imagenet / r.image_path.strip("/")
        else:
            return args.data_dir_flags / r.image_path.strip("/")
    df["image_path"] = df.apply(lambda r: _full_image_path(r), axis=1)

    return df


def tensors(split, dl, tensor_save_dir, audio_encoder, image_encoder, text_encoder, device, feat_token_id):
    print(f"Saving {split} tensors...")
    torch.set_grad_enabled(False)
    audio_reps = []
    image_reps = []
    text_reps = []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}

        # audio encoder
        x = audio_encoder(input_features=batch["audio_input_features"], attention_mask=batch["audio_attention_mask"])
        x = x["last_hidden_state"]
        x = x[:, 0, :]
        x = x.cpu()
        audio_reps.append(x)

        # image encoder
        x = image_encoder(pixel_values=batch["image_pixel_values"])
        x = x["last_hidden_state"]
        x = x[:, 0, :]
        x = x.cpu()
        image_reps.append(x)

        # text encoder
        x = text_encoder(input_ids=batch["text_input_ids"],
                        attention_mask=batch["text_attention_mask"])
        x = x["last_hidden_state"]
        x = x[torch.arange(x.shape[0]),
            (batch["text_input_ids"] == feat_token_id).int().argmax(dim=-1)]
        x = x.cpu()
        text_reps.append(x)

    audio_reps = torch.cat(audio_reps, dim=0)
    torch.save(audio_reps, tensor_save_dir + f'audio_{split}.pt')
    image_reps = torch.cat(image_reps, dim=0)
    torch.save(image_reps, tensor_save_dir + f'image_{split}.pt')
    text_reps = torch.cat(text_reps, dim=0)
    torch.save(text_reps, tensor_save_dir + f'text_{split}.pt')


if __name__ == '__main__':
    # options: "save_tensors", "split_train"
    do = "split_train"
    split = "train"

    if do == "save_tensors":
        args = parse_args_pretrain()

        # LOAD UP DATA
        audio_feat_extractor = WhisperFeatureExtractor.from_pretrained(args.audio_model_id)
        img_processor = CLIPImageProcessor.from_pretrained(args.image_model_id)
        txt_tokenizer = BertTokenizer.from_pretrained(args.text_model_id)
        feat_token_id = txt_tokenizer.sep_token_id
        num_workers = len(os.sched_getaffinity(0))

        dataset_path = args.train_dataset_path if split == "train" else args.val_dataset_path
        df = pd.read_csv(dataset_path)
        df = get_full_data_paths(df, args)
        ds = SymileDataset(df, audio_feat_extractor, img_processor)

        max_length = 108 if split == "train" else 512
        shuffle = True if split == "train" else False
        dl = DataLoader(ds, batch_size=200, shuffle=shuffle, num_workers=num_workers,
                        collate_fn=Collator(txt_tokenizer, max_length=max_length))

        tensor_save_dir = "/gpfs/scratch/as16583/tensors/"

        # LOAD UP MODELS
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audio_encoder = WhisperModel.from_pretrained(args.audio_model_id).encoder.to(device)
        image_encoder = CLIPVisionModel.from_pretrained(args.image_model_id).to(device)
        text_encoder = BertModel.from_pretrained(args.text_model_id).to(device)
        audio_encoder.eval()
        image_encoder.eval()
        text_encoder.eval()

        tensors(split, dl, tensor_save_dir, audio_encoder, image_encoder, text_encoder, device, feat_token_id)
    elif do == "split_train":
        split_train(split)