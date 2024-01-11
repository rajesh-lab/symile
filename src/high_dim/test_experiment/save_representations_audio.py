import os

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperFeatureExtractor, WhisperModel
from tqdm import tqdm

from args import parse_args_save_representations


class HighDimDataset(Dataset):
    def __init__(self, df, audio_feat_extractor):
        self.df = df

        self.audio_feat_extractor = audio_feat_extractor

    def __len__(self):
        return len(self.df)

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
        return torch.squeeze(audio.input_features)

    def __getitem__(self, idx):
        audio = self.get_audio(self.df.iloc[idx].audio_path)
        audio_filename = self.df.iloc[idx].audio_path.split("/")[-1].split(".")[0]

        return {"audio": audio,
                "audio_filename": audio_filename}


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.audio_feat_extractor = WhisperFeatureExtractor.from_pretrained(args.audio_model_id)

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))


class HighDimDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

    def setup(self, stage):
        df_train = pd.read_csv(self.args.data_dir / self.args.train_csv)
        self.ds_train = HighDimDataset(df_train, self.audio_feat_extractor)

        df_val = pd.read_csv(self.args.data_dir / self.args.val_csv)
        self.ds_val = HighDimDataset(df_val, self.audio_feat_extractor)

        df_test = pd.read_csv(self.args.data_dir / self.args.test_csv)
        self.ds_test = HighDimDataset(df_test, self.audio_feat_extractor)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          shuffle=True,
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


def load_audio_encoder(args, device):
    enc = WhisperModel.from_pretrained(args.audio_model_id).encoder.to(device)

    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()

    return enc


@torch.no_grad()
def save_representations(args, audio_encoder, dl, split, device):
    save_dir = args.save_dir / split / "audio"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # expects batch size of 1
    for ix, batch in enumerate(tqdm(dl)):
        batch["audio"] = batch["audio"].to(device)

        x = audio_encoder(batch["audio"])
        x = x["last_hidden_state"]
        x = torch.squeeze(x)
        x = x.cpu().numpy()
        np.save(save_dir / f"{batch['audio_filename'][0]}", x)


if __name__ == '__main__':
    args = parse_args_save_representations()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_encoder = load_audio_encoder(args, device)

    dm = HighDimDataModule(args)
    dm.prepare_data()

    dm.setup(stage="fit")
    print(f"Saving train tensors...")
    save_representations(args, audio_encoder, dm.train_dataloader(), "train", device)
    print(f"Saving val tensors...")
    save_representations(args, audio_encoder, dm.val_dataloader(), "val", device)

    dm.setup(stage="test")
    print(f"Saving test tensors...")
    save_representations(args, audio_encoder, dm.test_dataloader(), "test", device)