import json
import os
import time

import lightning.pytorch as pl
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import BertTokenizer, XLMRobertaTokenizer, MT5Tokenizer, \
                         CLIPVisionModel, CLIPImageProcessor, \
                         WhisperFeatureExtractor, WhisperModel, \
                         BertModel, XLMRobertaModel, MT5EncoderModel
from tqdm import tqdm

from args import parse_args_save_representations


class HighDimDataset(Dataset):
    def __init__(self, df, args, txt_tokenizer, img_processor, aud_feat_extractor, max_token_len):
        self.df = df
        self.args = args

        self.txt_tokenizer = txt_tokenizer
        self.img_processor = img_processor
        self.aud_feat_extractor = aud_feat_extractor

        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.df)

    def get_image(self, path):
        image = Image.open(self.args.imagenet_dir / path)
        image = self.img_processor(images=image, return_tensors="pt")
        return torch.squeeze(image.pixel_values)

    def get_audio(self, path):
        # downsample to 16kHz, as expected by Whisper, before passing to feature extractor
        waveform, sr = torchaudio.load(self.args.cv_dir / path)
        resampler = torchaudio.transforms.Resample(sr, self.aud_feat_extractor.sampling_rate)
        waveform = torch.squeeze(resampler(waveform))
        audio = self.aud_feat_extractor(
                        waveform,
                        return_attention_mask=True,
                        return_tensors="pt",
                        sampling_rate=self.aud_feat_extractor.sampling_rate,
                        do_normalize=True
                    )
        return torch.squeeze(audio.input_features)

    def __getitem__(self, idx):
        text = self.txt_tokenizer(text=self.df.iloc[idx].text,
                                  return_tensors="pt", padding="max_length",
                                  max_length=self.max_token_len)

        image = self.get_image(self.df.iloc[idx].image_path)

        audio = self.get_audio(self.df.iloc[idx].audio_path)

        return {"text": text,
                "image": image,
                "audio": audio,
                "cls": self.df.iloc[idx].cls,
                "cls_id": self.df.iloc[idx].cls_id,
                "lang": self.df.iloc[idx].lang,
                "idx": idx,
                "words": self.df.iloc[idx].text}


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.img_processor = CLIPImageProcessor.from_pretrained(args.image_model_id)
        self.aud_feat_extractor = WhisperFeatureExtractor.from_pretrained(args.audio_model_id)

        self.max_token_len = json.load(open(args.max_token_len_pt))

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def get_tokenizer(self):
        if self.args.text_model_id == "bert-base-multilingual-cased":
            self.txt_tokenizer = BertTokenizer.from_pretrained(self.args.text_model_id)
        elif self.args.text_model_id == "xlm-roberta-base" or self.args.text_model_id == "xlm-roberta-large":
            self.txt_tokenizer = XLMRobertaTokenizer.from_pretrained(self.args.text_model_id)
        elif self.args.text_model_id == "google/mt5-base" or self.args.text_model_id == "google/mt5-small" or self.args.text_model_id == "google/mt5-large" or self.args.text_model_id == "google/mt5-xxl":
            self.txt_tokenizer = MT5Tokenizer.from_pretrained(self.args.text_model_id)


class HighDimDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

    def setup(self, stage):
        self.get_tokenizer()

        df_train = pd.read_csv(self.args.data_dir / self.args.train_csv)
        self.ds_train = HighDimDataset(df_train, self.args, self.txt_tokenizer,
                                       self.img_processor, self.aud_feat_extractor,
                                       self.max_token_len["train"])

        df_val = pd.read_csv(self.args.data_dir / self.args.val_csv)
        self.ds_val = HighDimDataset(df_val, self.args, self.txt_tokenizer,
                                     self.img_processor, self.aud_feat_extractor,
                                     self.max_token_len["val"])

        df_test = pd.read_csv(self.args.data_dir / self.args.test_csv)
        self.ds_test = HighDimDataset(df_test, self.args, self.txt_tokenizer,
                                      self.img_processor, self.aud_feat_extractor,
                                      self.max_token_len["test"])

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last,
                          shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last,
                          shuffle=False)


def get_img_encoder(args):
    enc = CLIPVisionModel.from_pretrained(args.image_model_id)

    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()

    return enc


def get_aud_encoder(args):
    enc = WhisperModel.from_pretrained(args.audio_model_id).encoder

    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()

    return enc


def get_txt_encoder(args):
    if args.text_model_id == "bert-base-multilingual-cased":
            enc = BertModel.from_pretrained(args.text_model_id)
    elif args.text_model_id == "xlm-roberta-base" or args.text_model_id == "xlm-roberta-large":
            enc = XLMRobertaModel.from_pretrained(args.text_model_id)
    elif args.text_model_id == "google/mt5-base" or args.text_model_id == "google/mt5-small" or args.text_model_id == "google/mt5-large" or args.text_model_id == "google/mt5-xxl":
            enc = MT5EncoderModel.from_pretrained(args.text_model_id)

    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()

    return enc


@torch.no_grad()
def save_representations(args, dl, split, device):
    print(f"Saving {split} tensors...")

    save_dir = args.save_dir / split
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("save_dir: ", save_dir)

    img_encoder = get_img_encoder(args).to(device)
    aud_encoder = get_aud_encoder(args).to(device)

    text_input_ids = torch.empty(0)
    text_token_type_ids = torch.empty(0)
    text_attention_mask = torch.empty(0)
    image = torch.empty(0)
    audio = torch.empty(0)
    cls_id = torch.empty(0)
    idx = torch.empty(0)
    lang = []
    cls = []
    words = []

    for ix, batch in enumerate(tqdm(dl)):
        # TEXT
        text_input_ids = torch.cat((text_input_ids, batch["text"]["input_ids"].squeeze()), dim=0)
        text_attention_mask = torch.cat((text_attention_mask, batch["text"]["attention_mask"].squeeze()), dim=0)

        if args.text_model_id == "bert-base-multilingual-cased":
            text_token_type_ids = torch.cat((text_token_type_ids, batch["text"]["token_type_ids"].squeeze()), dim=0)

        # IMAGE
        x = img_encoder(pixel_values=batch["image"].to(device))
        x = x.pooler_output
        x = torch.squeeze(x)
        x = x.cpu()
        image = torch.cat((image, x), dim=0)

        # AUDIO
        x = aud_encoder(batch["audio"].to(device))
        x = x["last_hidden_state"]
        x = torch.squeeze(x)
        x = x.mean(dim=1)
        x = x.cpu()
        audio = torch.cat((audio, x), dim=0)

        # OTHER
        cls_id = torch.cat((cls_id, batch["cls_id"]), dim=0)
        idx = torch.cat((idx, batch["idx"]), dim=0)
        lang += batch["lang"]
        cls += batch["cls"]
        words += batch["words"]

    torch.save(text_input_ids, save_dir / f"text_input_ids_{split}.pt")
    torch.save(text_attention_mask, save_dir / f"text_attention_mask_{split}.pt")
    if args.text_model_id == "bert-base-multilingual-cased":
        torch.save(text_token_type_ids, save_dir / f"text_token_type_ids_{split}.pt")
    torch.save(image, save_dir / f"image_{split}.pt")
    torch.save(audio, save_dir / f"audio_{split}.pt")

    torch.save(cls_id, save_dir / f"cls_id_{split}.pt")
    torch.save(idx, save_dir / f"idx_{split}.pt")

    with open(f"{save_dir}/lang_{split}.txt", 'w') as f:
        f.writelines("\n".join(lang))

    with open(f"{save_dir}/cls_{split}.txt", 'w') as f:
        f.writelines("\n".join(cls))

    with open(f"{save_dir}/words_{split}.txt", 'w') as f:
        f.writelines("\n".join(words))


if __name__ == '__main__':
    start = time.time()

    args = parse_args_save_representations()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    dm = HighDimDataModule(args)
    dm.setup(stage="fit")

    train_split = str(args.train_csv).split(".")[0]

    if args.split_to_run == "all":
        save_representations(args, dm.train_dataloader(), train_split, device)
        save_representations(args, dm.val_dataloader(), "val", device)
        dm.setup(stage="test")
        save_representations(args, dm.test_dataloader(), "test", device)
    elif args.split_to_run == "train":
        save_representations(args, dm.train_dataloader(), train_split, device)
    elif args.split_to_run == "val":
        save_representations(args, dm.val_dataloader(), "val", device)
    elif args.split_to_run == "test":
        dm.setup(stage="test")
        save_representations(args, dm.test_dataloader(), "test", device)

    if args.split_to_run == "all" or args.split_to_run == "train":
        img_encoder = get_img_encoder(args)
        aud_encoder = get_aud_encoder(args)
        txt_encoder = get_txt_encoder(args)

        with open(args.data_dir / "metadata.json", 'w') as f:
            json.dump({
                "dataset_description": "text tokens saved (because text encoder will be finetuned); audio reps saved (because audio encoder will not be finetuned); image reps saved (because image encoder will not be finetuned)",
                "image_model_id": args.image_model_id,
                "audio_model_id": args.audio_model_id,
                "text_model_id": args.text_model_id,
                "image_enc_hidden_sz": img_encoder.config.hidden_size,
                "audio_enc_hidden_sz": aud_encoder.config.hidden_size,
                "text_enc_hidden_sz": txt_encoder.config.hidden_size,
            }, f, indent=4)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")