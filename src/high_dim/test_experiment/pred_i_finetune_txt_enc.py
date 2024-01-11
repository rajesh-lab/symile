from argparse import Namespace
from datetime import datetime
import os

import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, XLMRobertaTokenizer, \
                         BertModel, XLMRobertaModel, \
                         WhisperFeatureExtractor, WhisperModel, \
                         MT5EncoderModel, T5Tokenizer

from args import parse_args_main
from src.utils import l2_normalize


class HighDimDataset(Dataset):
    def __init__(self, df, audio_feat_extractor):
        self.df = df

        langs = sorted(df["lang"].unique())
        self.lang_embeddings = {value: idx for idx, value in enumerate(langs)}
        print("self.lang_embeddings:", self.lang_embeddings)

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
        return {"input_features": torch.squeeze(audio.input_features),
                "attention_mask": torch.squeeze(audio.attention_mask)}

    def __getitem__(self, idx):
        lang_embed = self.lang_embeddings[self.df.iloc[idx].lang]
        cls_id = self.df.iloc[idx].cls_id
        # audio = self.get_audio(self.df.iloc[idx].audio_path)
        text = self.df.iloc[idx].text

        return {"lang_embed": lang_embed, "cls_id": cls_id,
                # "audio": audio,
                "text": text, "idx": idx}


class Collator:
    """
    Custom collate function so that the text tokenizer can be called on a batch
    of text, which is then padded to the length of the longest sequence in the batch.
    """
    def __init__(self, txt_tokenizer):
        self.txt_tokenizer = txt_tokenizer
    def __call__(self, batch):
        # audio_input_features = torch.stack([s["audio"]["input_features"] for s in batch])
        # audio_attention_mask = torch.stack([s["audio"]["attention_mask"] for s in batch])

        text_list = [s["text"] for s in batch]
        text = self.txt_tokenizer(text=text_list, return_tensors="pt",
                                  padding=True, truncation=True)

        lang_embed = torch.tensor([s["lang_embed"] for s in batch])
        cls_id = torch.tensor([s["cls_id"] for s in batch])
        idx = torch.tensor([s["idx"] for s in batch])

        batched_data = {
            # "audio_input_features": audio_input_features,
            # "audio_attention_mask": audio_attention_mask,
            "text_input_ids": text["input_ids"],
            "text_attention_mask": text["attention_mask"],
            "lang_embed": lang_embed, "cls_id": cls_id, "idx": idx}

        return batched_data


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.audio_feat_extractor = WhisperFeatureExtractor.from_pretrained(args.audio_model_id)

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def text_tokenization(self):
        if self.args.text_model_id == "bert-base-multilingual-cased":
            self.txt_tokenizer = BertTokenizer.from_pretrained(self.args.text_model_id)
            if self.args.text_embedding == "eos":
                self.feat_token_id = self.txt_tokenizer.sep_token_id
            elif self.args.text_embedding == "bos":
                self.feat_token_id = self.txt_tokenizer.cls_token_id
        elif self.args.text_model_id == "xlm-roberta-base":
            self.txt_tokenizer = XLMRobertaTokenizer.from_pretrained(self.args.text_model_id)
            if self.args.text_embedding == "eos":
                self.feat_token_id = self.txt_tokenizer.eos_token_id
            elif self.args.text_embedding == "bos":
                self.feat_token_id = self.txt_tokenizer.bos_token_id
        elif self.args.text_model_id == "google/mt5-base" or self.args.text_model_id == "google/mt5-small":
            self.txt_tokenizer =tokenizer = T5Tokenizer.from_pretrained(self.args.text_model_id)
            if self.args.text_embedding == "eos":
                self.feat_token_id = self.txt_tokenizer.eos_token_id
            elif self.args.text_embedding == "bos":
                self.feat_token_id = self.txt_tokenizer.bos_token_id


class HighDimDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

    def setup(self, stage):
        self.text_tokenization()

        df_train = pd.read_csv(self.args.data_dir / self.args.train_csv)
        self.ds_train = HighDimDataset(df_train, self.audio_feat_extractor)

        df_val = pd.read_csv(self.args.data_dir / self.args.val_csv)
        self.ds_val = HighDimDataset(df_val, self.audio_feat_extractor)

        df_test = pd.read_csv(self.args.data_dir / self.args.test_csv)
        self.ds_test = HighDimDataset(df_test, self.audio_feat_extractor)

    # YODODODOOTTOOODODOOO
        # YODODODOOTTOOODODOOO# YODODODOOTTOOODODOOO
    # YODODODOOTTOOODODOOO

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer),
                          drop_last=self.args.drop_last)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer),
                          drop_last=self.args.drop_last)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer),
                          drop_last=self.args.drop_last)


class TextEncoder(nn.Module):
    def __init__(self, model_id, d, feat_token_id):
        super().__init__()
        self.feat_token_id = feat_token_id
        if model_id == "bert-base-multilingual-cased":
            self.encoder = BertModel.from_pretrained(model_id)
        elif model_id == "xlm-roberta-base":
            self.encoder = XLMRobertaModel.from_pretrained(model_id)

    def forward(self, x):
        """
        If not using precomputed tensors:
            Args:
                x (dict): keys are "input_ids" and "attention_mask":
                    input_ids (torch.Tensor): shape (batch_sz, len_longest_seq)
                    attention_mask (torch.Tensor): shape (batch_sz, len_longest_seq)
            Returns:
                x (torch.Tensor): shape (batch_sz, d)
        """
        x_arg = x
        x = self.encoder(input_ids=x_arg["input_ids"],
                            attention_mask=x_arg["attention_mask"])
        x = x["last_hidden_state"] # shape (batch_sz, len_longest_seq, d)

        # take features from EOS or BOS embedding. x has shape (b, l, d).
        # argmax returns first index of feat_token_id in case pad_token_id is
        # equal to feat_token_id.
        x = x[torch.arange(x.shape[0]),
            (x_arg["input_ids"] == self.feat_token_id).int().argmax(dim=-1)] # (b, d)
        return x


class SSLModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()

        self.args = Namespace(**args)

        self.text_encoder = TextEncoder(self.args.text_model_id, self.args.d,
                                        self.args.feat_token_id)

        self.lang_embedding = nn.Embedding(5, self.args.d).to(self.device)
        # self.fc1 = nn.Linear(self.args.d, self.args.hidden_layer_d)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(self.args.hidden_layer_d, 50)
        self.fc1 = nn.Linear(self.args.d, 50)

    def forward(self, x, verbose=False):
        r_t = self.text_encoder({"input_ids": x["text_input_ids"],
                                 "attention_mask": x["text_attention_mask"]})

        lang_embed = self.lang_embedding(x["lang_embed"].to(torch.int))

        [r_t, lang_embed] = l2_normalize([r_t, lang_embed])

        print("r_t norm:", r_t.norm())
        print("lang_embed norm:", lang_embed.norm())

        # if self.current_epoch < 2:
        #     r_t = r_t.detach()

        if verbose:
            print("r_t: ", r_t.abs().mean())

        # todo CHECK SHAPES

        input_tensor = r_t * lang_embed
        breakpoint()

        # x = self.fc1(input_tensor)
        # x = self.relu(x)
        # x = self.fc2(x)

        x = self.fc1(input_tensor)

        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def on_after_backward(self):
        norm = torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), float('inf'))
        print("norm: ", norm)
        self.log("grad_norm", norm, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

    def _shared_step(self, batch, batch_idx):

        v = (batch_idx%10==0)
        logits = self(batch, verbose=False) # (b, num_classes)
        labels = batch["cls_id"].long() # (b)

        if v:
            print("logits: ", logits.abs().mean())

        pred = torch.argmax(logits, dim=1)
        m = torch.max(logits, dim=1)
        print("pred: ", pred)
        print("m: ", m)

        loss = nn.CrossEntropyLoss()
        return loss(logits, labels)

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)

        # x = [p for p in self.text_encoder.encoder.encoder.parameters()][0]
        # print(x)

        self.log("train_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch) # (b, 1000)
        labels = batch["cls_id"].long() # (b)

        pred = torch.argmax(logits, dim=1)

        acc = torch.sum(pred == labels) / len(labels)

        self.log("test_accuracy", acc, sync_dist=True, prog_bar=True)


def main(args, trainer):
    dm = HighDimDataModule(args)
    dm.setup(stage="fit")
    args.feat_token_id = dm.feat_token_id

    model = SSLModel(**vars(args))
    trainer.fit(model, datamodule=dm)

    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == '__main__':
    os.environ['WANDB_CACHE_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'
    os.environ['WANDB_CONFIG_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'
    os.environ['WANDB_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'
    os.environ['WANDB_DATA_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'

    args = parse_args_main()

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    save_dir = args.ckpt_save_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(args.ckpt_save_dir):
        os.makedirs(args.ckpt_save_dir)
    os.mkdir(save_dir)
    setattr(args, "save_dir", save_dir)
    print("\nSaving to: ", save_dir)

    if args.wandb:
        logger = WandbLogger(project="symile", log_model="all", save_dir=args.ckpt_save_dir)
    else:
        logger = False

    checkpoint_callback = ModelCheckpoint(dirpath=save_dir,
                                          every_n_epochs = 1,
                                          filename="{epoch}-{val_loss:.2f}",
                                          mode="min",
                                          monitor="val_loss")
    early_stopping_callback = EarlyStopping(monitor="val_loss",
                                            mode="min",
                                            patience=args.early_stopping_patience)
    if args.early_stopping:
        callbacks = [checkpoint_callback, early_stopping_callback]
    else:
        callbacks = [checkpoint_callback]

    trainer = Trainer(
        callbacks=callbacks,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.use_seed,
        enable_progress_bar=True,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=1,
        logger=logger,
        max_epochs=args.epochs,
        num_sanity_val_steps=0
    )

    main(args, trainer)