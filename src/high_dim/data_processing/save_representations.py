import os

import lightning.pytorch as pl
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import BertTokenizer, XLMRobertaTokenizer, T5Tokenizer, \
                         CLIPVisionModel, CLIPImageProcessor, \
                         WhisperFeatureExtractor, WhisperModel
from tqdm import tqdm

from args import parse_args_save_representations


class HighDimDataset(Dataset):
    def __init__(self, df, txt_tokenizer, img_processor, aud_feat_extractor):
        self.df = df

        self.txt_tokenizer = txt_tokenizer
        self.img_processor = img_processor
        self.aud_feat_extractor = aud_feat_extractor

        self.max_length = self.get_max_length()

    def __len__(self):
        return len(self.df)

    def get_max_length(self):
        text = self.txt_tokenizer(text=self.df.text.tolist(), return_tensors="pt", padding=True)
        return text["input_ids"].shape[1]

    def get_image(self, path):
        image = Image.open(path)
        image = self.img_processor(images=image, return_tensors="pt")
        return torch.squeeze(image.pixel_values)

    def get_audio(self, path):
        # downsample to 16kHz, as expected by Whisper, before passing to feature extractor
        waveform, sr = torchaudio.load(path)
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
                                  max_length=self.max_length)

        image = self.get_image(self.df.iloc[idx].image_path)

        audio = self.get_audio(self.df.iloc[idx].audio_path)

        return {"text": text,
                "image": image,
                "audio": audio,
                "cls": self.df.iloc[idx].cls,
                "cls_id": self.df.iloc[idx].cls_id,
                "lang": self.df.iloc[idx].lang,
                "idx": idx}


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.img_processor = CLIPImageProcessor.from_pretrained(args.image_model_id)
        self.aud_feat_extractor = WhisperFeatureExtractor.from_pretrained(args.audio_model_id)

        # from max_num_worker_suggest in DataLoader docs
        # self.num_workers = len(os.sched_getaffinity(0))
        self.num_workers = 0

    def get_tokenizer(self):
        if self.args.text_model_id == "bert-base-multilingual-cased":
            self.txt_tokenizer = BertTokenizer.from_pretrained(self.args.text_model_id)
        elif self.args.text_model_id == "xlm-roberta-base":
            self.txt_tokenizer = XLMRobertaTokenizer.from_pretrained(self.args.text_model_id)
        elif self.args.text_model_id == "google/mt5-base" or self.args.text_model_id == "google/mt5-small":
            self.txt_tokenizer = T5Tokenizer.from_pretrained(self.args.text_model_id)


class HighDimDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

    def setup(self, stage):
        self.get_tokenizer()

        df_train = pd.read_csv(self.args.data_dir / self.args.train_csv)
        self.ds_train = HighDimDataset(df_train, self.txt_tokenizer,
                                       self.img_processor, self.aud_feat_extractor)

        df_val = pd.read_csv(self.args.data_dir / self.args.val_csv)
        self.ds_val = HighDimDataset(df_val, self.txt_tokenizer,
                                     self.img_processor, self.aud_feat_extractor)

        df_test = pd.read_csv(self.args.data_dir / self.args.test_csv)
        self.ds_test = HighDimDataset(df_test, self.txt_tokenizer,
                                      self.img_processor, self.aud_feat_extractor)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)


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


@torch.no_grad()
def save_representations(args, dl, split, device):
    save_dir = args.save_dir / split
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

    for ix, batch in enumerate(tqdm(dl)):
        # TEXT
        text_input_ids = torch.cat((text_input_ids, batch["text"]["input_ids"].squeeze()), dim=0)
        text_token_type_ids = torch.cat((text_token_type_ids, batch["text"]["token_type_ids"].squeeze()), dim=0)
        text_attention_mask = torch.cat((text_attention_mask, batch["text"]["attention_mask"].squeeze()), dim=0)

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

    torch.save(text_input_ids, save_dir / f"text_input_ids_{split}.pt")
    torch.save(text_token_type_ids, save_dir / f"text_token_type_ids_{split}.pt")
    torch.save(text_attention_mask, save_dir / f"text_attention_mask_{split}.pt")
    torch.save(image, save_dir / f"image_{split}.pt")
    torch.save(audio, save_dir / f"audio_{split}.pt")
    torch.save(cls_id, save_dir / f"cls_id_{split}.pt")
    torch.save(idx, save_dir / f"idx_{split}.pt")

    with open(f"{save_dir}/lang_{split}.txt", 'w') as f:
        f.writelines("\n".join(lang))

    with open(f"{save_dir}/cls_{split}.txt", 'w') as f:
        f.writelines("\n".join(cls))


if __name__ == '__main__':
    args = parse_args_save_representations()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = HighDimDataModule(args)
    dm.setup(stage="fit")

    print(f"Saving train tensors...")
    save_representations(args, dm.train_dataloader(), "train", device)
    print(f"Saving val tensors...")
    save_representations(args, dm.val_dataloader(), "val", device)

    dm.setup(stage="test")
    print(f"Saving test tensors...")
    save_representations(args, dm.test_dataloader(), "test", device)