import os
from pathlib import Path
import pandas as pd

import numpy as np
from PIL import Image
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, CLIPImageProcessor, \
                         WhisperFeatureExtractor, XLMRobertaTokenizer
from transformers import BertModel, CLIPVisionModel, WhisperModel, XLMRobertaModel
from tqdm import tqdm

from args import parse_args_pretrain
from constants import *

# def get_class_mappings(mapping_path):
#     """
#     Create class mapping dataframe from ImageNet synset mapping file.

#     Args:
#         mapping_path (Path): Path to ImageNet synset mapping file (must be .txt).
#     Returns:
#         (pd.DataFrame): columns are `class_id` and `class_name`.
#     """
#     def _synsetmapping_to_name(synset_str):
#         synset_str = synset_str.split(" ")[1:]
#         return " ".join(synset_str).split(",")[0]

#     class_mapping = pd.read_csv(mapping_path, sep="\t", names=["synset"])
#     class_mapping["class_id"] = class_mapping.synset.apply(lambda x: x.split(" ")[0])
#     class_mapping["class_name"] = class_mapping.synset.apply(_synsetmapping_to_name)
#     return class_mapping[["class_id", "class_name"]]


class SymileDataset(Dataset):
    def __init__(self, df, audio_feat_extractor, img_processor):
        self.df = df
        self.audio_feat_extractor = audio_feat_extractor
        self.img_processor = img_processor

    def __len__(self):
        return len(self.df)

    # def add_classification_labels(self, df):
    #     mapping = get_class_mappings(Path("/gpfs/data/ranganathlab/adriel/imagenet/LOC_synset_mapping.txt"))
    #     def _get_language(r):
    #         if r.template == 1:
    #             return str(r.audio_path).split("/")[-1].split("_")[0]
    #         elif r.template == 2:
    #             return FLAGFILE2ISO[str(r.image_path).split("/")[-1]]
    #     def _get_object(r, mapping):
    #         if r.template == 1:
    #             class_id = str(r.image_path).split("/")[-2]
    #             class_clf = mapping[mapping["class_id"] == class_id].index.item()
    #             return class_clf
    #         elif r.template == 2:
    #             return -1
    #     df["language"] = df.apply(lambda r: _get_language(r), axis=1)
    #     df["object"] = df.apply(lambda r: _get_object(r, mapping), axis=1)
    #     return df

    def get_audio(self, path):
        # downsample to 16kHz, as expected by Whisper, before passing to feature extractor
        waveform, sr = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sr, self.audio_feat_extractor.sampling_rate)
        waveform = torch.squeeze(resampler(waveform))
        audio = self.audio_feat_extractor(
                        waveform,
                        return_attention_mask=True,
                        return_tensors="pt",
                        sampling_rate=self.audio_feat_extractor.sampling_rate,
                        do_normalize=True
                    )
        return {"input_features": torch.squeeze(audio.input_features),
                "attention_mask": torch.squeeze(audio.attention_mask)}

    def get_image(self, path):
        image = Image.open(path)
        image = self.img_processor(images=image, return_tensors="pt")
        return {"pixel_values": torch.squeeze(image.pixel_values)}

    def __getitem__(self, idx):
        """
        Returns:
            (dict): containing the following key-value pairs:
                - audio: (dict) whose key-value pairs are
                    (input_features: torch.Tensor of shape (80, 3000)) and
                    (attention_mask: torch.Tensor of shape (3000)).
                - image: (dict) whose key-value pairs are
                    (pixel_values: torch.Tensor of shape (3, 224, 224)).
                - text: (str) with data sample text.
                - template: (int) with data sample template number.
                - idx: (int) unique identifier for data sample.
                - support (optional): (int) 1 if data sample is in support, 0 otherwise.
        """
        audio = self.get_audio(self.df.iloc[idx].audio_path)
        image = self.get_image(self.df.iloc[idx].image_path)
        text = self.df.iloc[idx].text
        in_support = self.df.iloc[idx].in_support
        # lang_cls = self.df.iloc[idx].lang_cls
        # object_cls = self.df.iloc[idx].object_cls

        item_dict = {"audio": audio, "image": image, "text": text,
                    "in_support": in_support}
                    #  "lang_cls": lang_cls, "object_cls": object_cls}

        return item_dict

def count_max_length(txt_tokenizer):
    # root_dir = "/gpfs/scratch/as16583/tensors_100_classes/"
    # files = ["train.csv", "val.csv", "test.csv"]
    # for f in files:
    #     print("counting max length for ", f, "...\n")
    #     df = pd.read_csv(root_dir + f)
    #     text = txt_tokenizer(text=df.text.tolist(), return_tensors="pt", padding=True)
    #     max_length = text["input_ids"].shape[1]
    #     print("max length is: ", max_length, "\n")

    f = "/gpfs/scratch/as16583/symile/src/symile_m3/data/sources/support_test_temp1_insupport1.csv"
    print("counting max length for ", f, "...\n")
    df = pd.read_csv(f)
    text = txt_tokenizer(text=df.text.tolist(), return_tensors="pt", padding=True)
    max_length = text["input_ids"].shape[1]
    print("max length is: ", max_length, "\n")

def split_train(split):
    read_dir = Path("/gpfs/scratch/as16583/xlm_whispersmall_tensors/pretrain")
    audio = torch.load(read_dir / f"audio_{split}.pt")
    image = torch.load(read_dir / f"image_{split}.pt")
    text = torch.load(read_dir / f"text_{split}.pt")
    language = torch.load(read_dir / f"language_{split}.pt")
    object_clf = torch.load(read_dir / f"object_{split}.pt")
    template = torch.load(read_dir / f"template_{split}.pt")

    permutation = torch.randperm(len(audio))
    audio = audio[permutation]
    image = image[permutation]
    text = text[permutation]
    language = language[permutation]
    object_clf = object_clf[permutation]
    template = template[permutation]

    save_dir = Path("/gpfs/scratch/as16583/xlm_whispersmall_tensors/pretrain")

    audio_train = audio[0:10000]
    audio_test = audio[10000:]
    torch.save(audio_train, save_dir / f"audio_train.pt")
    torch.save(audio_test, save_dir / f"audio_test.pt")

    image_train = image[0:10000]
    image_test = image[10000:]
    torch.save(image_train, save_dir / f"image_train.pt")
    torch.save(image_test, save_dir / f"image_test.pt")

    text_train = text[0:10000]
    text_test = text[10000:]
    torch.save(text_train, save_dir / f"text_train.pt")
    torch.save(text_test, save_dir / f"text_test.pt")

    language_train = language[0:10000]
    language_test = language[10000:]
    torch.save(language_train, save_dir / f"language_train.pt")
    torch.save(language_test, save_dir / f"language_test.pt")

    object_train = object_clf[0:10000]
    object_test = object_clf[10000:]
    torch.save(object_train, save_dir / f"object_train.pt")
    torch.save(object_test, save_dir / f"object_test.pt")

    template_train = template[0:10000]
    template_test = template[10000:]
    torch.save(template_train, save_dir / f"template_train.pt")
    torch.save(template_test, save_dir / f"template_test.pt")


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

        # templates = torch.Tensor([s["template"] for s in batch])
        # idx = torch.Tensor([s["idx"] for s in batch])
        # lang_cls = torch.Tensor([s["lang_cls"] for s in batch])
        # object_cls = torch.Tensor([s["object_cls"] for s in batch])

        batched_data = {"audio_input_features": audio_input_features,
                        "audio_attention_mask": audio_attention_mask,
                        "image_pixel_values": image_pixel_values,
                        "text_input_ids": text["input_ids"],
                        "text_attention_mask": text["attention_mask"],
                        "in_support": torch.Tensor([s["in_support"] for s in batch])}
                        # "templates": templates,
                        # "idx": idx,
                        # "lang_cls": lang_cls,
                        # "object_cls": object_cls}

        return batched_data


def get_full_data_paths(df, args):
    def _full_audio_path(r):
        return args.data_dir_generated_audio / r.audio_path.strip("/")
    df["audio_path"] = df.apply(lambda r: _full_audio_path(r), axis=1)

    # def _full_image_path(r):
    #     if r.template == 1:
    #         return args.data_dir_imagenet / r.image_path.strip("/")
    #     else:
    #         return args.data_dir_flags / r.image_path.strip("/")
    def _full_image_path(r):
        return args.data_dir_imagenet / r.image_path.strip("/")
    df["image_path"] = df.apply(lambda r: _full_image_path(r), axis=1)

    return df


def tensors(split, dl, tensor_save_dir, audio_encoder, image_encoder, text_encoder, device, feat_token_id):
    print(f"Saving {split} tensors...")
    torch.set_grad_enabled(False)
    audio_reps = []
    image_reps = []
    text_reps = []
    # language_reps = []
    # object_reps = []
    in_support = []
    dl_loop = tqdm(dl)
    for ix, batch in enumerate(dl_loop):
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

        # language and object classifications and templates
        # language_reps.append(batch["lang_cls"].cpu())
        # object_reps.append(batch["object_cls"].cpu())
        in_support.append(batch["in_support"].cpu())

    audio_reps = torch.cat(audio_reps, dim=0)
    torch.save(audio_reps, tensor_save_dir + f'audio_{split}.pt')
    image_reps = torch.cat(image_reps, dim=0)
    torch.save(image_reps, tensor_save_dir + f'image_{split}.pt')
    text_reps = torch.cat(text_reps, dim=0)
    torch.save(text_reps, tensor_save_dir + f'text_{split}.pt')
    # language_reps = torch.cat(language_reps, dim=0)
    # torch.save(language_reps, tensor_save_dir + f'language_{split}.pt')
    # object_reps = torch.cat(object_reps, dim=0)
    # torch.save(object_reps, tensor_save_dir + f'object_{split}.pt')
    in_support = torch.cat(in_support, dim=0)
    torch.save(in_support, tensor_save_dir + f'in_support_{split}.pt')


if __name__ == '__main__':
    # options: "save_tensors", "split_train", "count_max_length"
    do = "save_tensors"
    file = "support_test_temp1_insupport1_dropduptext.csv"
    split = "test"
    tensor_save_dir = "/gpfs/scratch/as16583/tensors_100_classes/support/insupport_dropduptext/"
    audio_model_id = "openai/whisper-small"
    image_model_id = "openai/clip-vit-base-patch16"
    text_model_id = "xlm-roberta-base"

    print("split is: ", split)

    args = parse_args_pretrain()

    if text_model_id == "bert-base-multilingual-cased":
        txt_tokenizer = BertTokenizer.from_pretrained(text_model_id)
        feat_token_id = txt_tokenizer.sep_token_id
        max_lengths = {"pretrain_train.csv": 29, "pretrain_val.csv": 23,
                    "zeroshot_test.csv": 23, "support_train.csv": 29,
                    "support_val.csv": 22, "support_test.csv": 27}
    elif text_model_id == "xlm-roberta-base":
        txt_tokenizer = XLMRobertaTokenizer.from_pretrained(text_model_id)
        feat_token_id = txt_tokenizer.eos_token_id
        max_lengths = {"pretrain_train.csv": 21, "pretrain_val.csv": 22,
                    "zeroshot_test.csv": 22, "support_train.csv": 22,
                    "support_val.csv": 17, "support_test.csv": 22}

    if do == "save_tensors":

        # LOAD UP DATA
        audio_feat_extractor = WhisperFeatureExtractor.from_pretrained(audio_model_id)
        img_processor = CLIPImageProcessor.from_pretrained(image_model_id)
        num_workers = len(os.sched_getaffinity(0))

        dataset_path = "/gpfs/scratch/as16583/symile/src/symile_m3/data/sources/" + file
        # dataset_path = "/gpfs/scratch/as16583/tensors_100_classes/" + file

        df = pd.read_csv(dataset_path)
        df = get_full_data_paths(df, args)
        ds = SymileDataset(df, audio_feat_extractor, img_processor)

        shuffle = True if split == "train" else False
        dl = DataLoader(ds, batch_size=200, shuffle=shuffle, num_workers=num_workers,
                        collate_fn=Collator(txt_tokenizer, max_length=22)) #22 for support_test_temp1_insupport1

        # LOAD UP MODELS
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audio_encoder = WhisperModel.from_pretrained(audio_model_id).encoder.to(device)
        image_encoder = CLIPVisionModel.from_pretrained(image_model_id).to(device)
        if text_model_id == "bert-base-multilingual-cased":
            text_encoder = BertModel.from_pretrained(text_model_id).to(device)
        elif text_model_id == "xlm-roberta-base":
            text_encoder = XLMRobertaModel.from_pretrained(text_model_id).to(device)
        audio_encoder.eval()
        image_encoder.eval()
        text_encoder.eval()

        tensors(split, dl, tensor_save_dir, audio_encoder, image_encoder, text_encoder, device, feat_token_id)
    elif do == "split_train":
        split_train(split)
    elif do == "count_max_length":
        count_max_length(txt_tokenizer)