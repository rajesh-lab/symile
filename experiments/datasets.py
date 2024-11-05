import os

import numpy as np
import lightning.pytorch as pl
from scipy.stats import bernoulli
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer

from constants import MISSING_TOKEN
from utils import get_language_constant


##############
# Binary XOR #
##############


class BinaryXORDataset(Dataset):
    """
    Generate n samples of data (v_a, v_b, v_c) according to the below.

    i ~ Bernoulli(p_hat)
    dim(v_a) = dim(v_b) = dim(v_c) = d_v
    v_a[j], v_b[j] ~ Bernoulli(0.5)
    v_c[j] = (v_a[j] XOR v_b[j])^i * v_a[j]^(1-i)
    """
    def __init__(self, d_v, n, p_hat):
        """
        Args:
            d_v (int): dimensionality for each of the vectors v_a, v_b, v_c.
            n (int): number of data samples to generate.
            p_hat (float): Bernoulli distribution parameter for i.
        """
        self.v_a, self.v_b, self.v_c = self.generate_data(d_v, n, p_hat)

    def generate_data(self, d_v, n, p_hat):
        """
        Returns:
            v_a, v_b, v_c: each is an torch.Tensor of size (n, d_v).
        """
        v_a = bernoulli.rvs(0.5, size=(n, d_v))
        v_b = bernoulli.rvs(0.5, size=(n, d_v))
        i = bernoulli.rvs(p_hat, size=n)

        xor = np.bitwise_xor(v_a, v_b)

        if d_v == 1:
            i = np.expand_dims(i, axis=1)
            v_c = np.where(i, xor, v_a)
        else: # d_v > 1
            c_columns = []
            for j in range(d_v):
                c_columns.append(np.where(i, xor[:, j], v_a[:, j]))
            v_c = np.stack(c_columns, axis=1)

        v_a = torch.from_numpy(v_a).to(torch.float32)
        v_b = torch.from_numpy(v_b).to(torch.float32)
        v_c = torch.from_numpy(v_c).to(torch.float32)

        assert v_a.shape == v_b.shape == v_c.shape, \
            "Random variables must be the same shape"
        for arr in (v_a, v_b, v_c):
            assert torch.all((arr == 0) | (arr == 1)), "Random variables must be 0 or 1."
        assert v_a.shape[1] == d_v, "Vectors must have dimension d_v."

        return v_a, v_b, v_c

    def __len__(self):
        """
        Compute length of the dataset.

        Returns:
            (int): dataset size.
        """
        return len(self.v_a)

    def __getitem__(self, idx):
        """
        Index into the dataset.

        Args:
            idx (int): index of data sample to retrieve.
        Returns
            v_a, v_b, v_c (tuple): each of v_a, v_b, v_c is a torch.Tensor of size d_v.
        """
        v_a = self.v_a[idx, :]
        v_b = self.v_b[idx, :]
        v_c = self.v_c[idx, :]
        return v_a, v_b, v_c


class BinaryXORDataModule(pl.LightningDataModule):
    def __init__(self, args):
        """
        Initialize LightningDataModule for the binary XOR dataset.

        Args:
            args (Namespace): contains arguments for dataset configuration and training.
        """
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def setup(self, stage):
        self.ds_train = BinaryXORDataset(self.args.d_v, self.args.train_n, self.args.p_hat)
        self.ds_val = BinaryXORDataset(self.args.d_v, self.args.val_n, self.args.p_hat)
        self.ds_test = BinaryXORDataset(self.args.d_v, self.args.test_n, self.args.p_hat)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=True)

    def resample_test_set(self):
        self.ds_test = BinaryXORDataset(self.args.d_v, self.args.test_n, self.args.p_hat)


#############
# Symile-M3 #
#############


class SymileM3Dataset(Dataset):
    """
    Symile-M3 Dataset
    """
    def __init__(self, args, split, txt_tokenizer=None):
        """
        Loads data for `split` from disk.

        Args:
            args (Namespace): contains arguments for dataset configuration.
            split (str): dataset split to use (`train`, `val`, or `test`).
            txt_tokenizer (Tokenizer, optional): Tokenizer for processing text
                data; txt_tokenizer is not None when args.missingness is True.
        """
        self.args = args
        self.split = split
        self.txt_tokenizer = txt_tokenizer

        self.split_dir = self.args.data_dir / f"{split}"

        self.text_input_ids = torch.load(self.split_dir / f"text_input_ids_{split}.pt").long()
        self.text_attention_mask = torch.load(self.split_dir / f"text_attention_mask_{split}.pt")
        self.max_token_len = self.text_input_ids.shape[1]

        self.image = torch.load(self.split_dir / f"image_{split}.pt")
        self.image_mean = torch.mean(self.image, dim=0)

        self.audio = torch.load(self.split_dir / f"audio_{split}.pt")
        self.audio_mean = torch.mean(self.audio, dim=0)

        self.cls_id = torch.load(self.split_dir / f"cls_id_{split}.pt")
        self.idx = torch.load(self.split_dir / f"idx_{split}.pt")

        with open(self.split_dir / f"lang_{split}.txt", "r") as f:
            self.lang = f.read().splitlines()

        self.languages = get_language_constant(self.args.num_langs)

        # If running an experiment that includes missingness, load the missingness tensors.
        # The missingness tensors are only used during training and validation.
        if getattr(self.args, "missingness", False) and self.split != "test":
            if self.args.missingness_prob == 0.5:
                missingness_prob_str = "50"
            elif self.args.missingness_prob == 0.6:
                missingness_prob_str = "60"
            elif self.args.missingness_prob == 0.65:
                missingness_prob_str = "65"
            elif self.args.missingness_prob == 0.7:
                missingness_prob_str = "70"
            elif self.args.missingness_prob == 0.75:
                missingness_prob_str = "75"
            else:
                raise ValueError("Missingness probability not supported.")
            self.text_missingness = torch.load(self.split_dir / f"text_missingness_prob{missingness_prob_str}_{split}.pt")
            self.image_missingness = torch.load(self.split_dir / f"image_missingness_prob{missingness_prob_str}_{split}.pt")
            self.audio_missingness = torch.load(self.split_dir / f"audio_missingness_prob{missingness_prob_str}_{split}.pt")

    def __len__(self):
        return len(self.image)

    def get_missingness_text(self):
        """
        Get a tokenized representation for MISSING_TOKEN.

        Returns:
            dict: with keys "input_ids" and "attention_mask", whose values are
                  Torch.tensors with shape (self.max_token_len).
        """
        encoded_inputs = self.txt_tokenizer(text=MISSING_TOKEN,
                                            return_tensors="pt",
                                            padding="max_length",
                                            max_length=self.max_token_len)
        encoded_inputs["input_ids"] = torch.squeeze(encoded_inputs["input_ids"], dim=0)
        encoded_inputs["attention_mask"] = torch.squeeze(encoded_inputs["attention_mask"], dim=0)
        encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"].to(torch.float32)
        return encoded_inputs

    def __getitem__(self, idx):
        """
        Indexes into the dataset.

        If running a missingness experiment, the text, image, and audio data may be missing.
        If the text data is missing, the text data is replaced with a tokenized representation of MISSING_TOKEN.
        If the image or audio data is missing, it is replaced with the mean image or mean audio computed from
        the training set, respectively.

        Args:
            idx (int): Index of data sample to retrieve.

        Returns:
            dict: A dictionary containing the following key-value pairs:
                - text (dict): Dictionary with keys "input_ids" and "attention_mask".
                - image (torch.Tensor): Tensor with image data.
                - audio (torch.Tensor): Tensor with audio data.
                - cls_id (torch.float32): Tensor containing the class id for the sample
                    (as determined by the image class name).
                - idx (torch.float32): Tensor containing the unique identifier for the sample.
                - lang_id (int): Integer representing the language id for the sample.
                - text_missing (torch.int32): Integer indicating whether the text data is observed (0) or missing (1).
                - image_missing (torch.int32): Integer indicating whether the image data is observed (0) or missing (1).
                - audio_missing (torch.int32): Integer indicating whether the audio data is observed (0) or missing (1).
                - all_observed (int): Integer indicating whether all modalities are observed (1) or if some modalities
                    are missing (0).
        """
        text = {"input_ids": self.text_input_ids[idx],
                "attention_mask": self.text_attention_mask[idx]}
        image = self.image[idx]
        audio = self.audio[idx]

        text_missing = 0
        image_missing = 0
        audio_missing = 0

        # If running an experiment that includes missingness, load the missingness tensors.
        # The missingness tensors are only used during training and validation.
        if getattr(self.args, "missingness", False) and self.split != "test":
            text_missing = self.text_missingness[idx]
            image_missing = self.image_missingness[idx]
            audio_missing = self.audio_missingness[idx]

            if text_missing == 1:
                text = self.get_missingness_text()

            if image_missing == 1:
                image = self.image_mean

            if audio_missing == 1:
                audio = self.audio_mean

        if (text_missing == 0) and (image_missing == 0) and (audio_missing == 0):
            all_observed = 1
        else:
            all_observed = 0

        return {"text": text,
                "image": image,
                "audio": audio,
                "cls_id": self.cls_id[idx],
                "idx": self.idx[idx],
                "lang_id": self.languages.index(self.lang[idx]),
                "text_missing": text_missing,
                "image_missing": image_missing,
                "audio_missing": audio_missing,
                "all_observed": all_observed}


class SymileM3DataModule(pl.LightningDataModule):
    def __init__(self, args):
        """
        Initialize LightningDataModule for the Symile-M3 dataset.

        Args:
            args (Namespace): contains arguments for dataset configuration and training.
        """
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

        self.txt_tokenizer = None

        # Text tokenizer needed for missingness experiments to get a tokenized
        # representation for MISSING_TOKEN.
        if self.args.missingness:

            self.txt_tokenizer = AutoTokenizer.from_pretrained(args.text_model_id)

            if MISSING_TOKEN not in self.txt_tokenizer.get_vocab():
                self.txt_tokenizer.add_tokens([MISSING_TOKEN])

                # update the tokenizer length to include the missing token
                # (needed during model initialization)
                self.tokenizer_len = len(self.txt_tokenizer)
            else:
                raise ValueError("MISSING_TOKEN already exists in the tokenizer vocab.")

    def setup(self, stage):
        self.ds_train = SymileM3Dataset(self.args, "train", self.txt_tokenizer)
        self.ds_val = SymileM3Dataset(self.args, "val", self.txt_tokenizer)
        self.ds_test = SymileM3Dataset(self.args, "test", self.txt_tokenizer)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False)


################
# Symile-MIMIC #
################


class SymileMIMICRetrievalDataset(Dataset):
    def __init__(self, args, split):
        self.cxr = torch.load(args.data_dir / f"{split}/cxr_{split}.pt")
        self.ecg = torch.load(args.data_dir / f"{split}/ecg_{split}.pt")
        self.labs_percentiles = torch.load(args.data_dir / f"{split}/labs_percentiles_{split}.pt")
        self.labs_missingness = torch.load(args.data_dir / f"{split}/labs_missingness_{split}.pt")
        self.hadm_id = torch.load(args.data_dir / f"{split}/hadm_id_{split}.pt")
        self.label_hadm_id = torch.load(args.data_dir / f"{split}/label_hadm_id_{split}.pt")
        self.label = torch.load(args.data_dir / f"{split}/label_{split}.pt")

    def __len__(self):
        return len(self.ecg)

    def __getitem__(self, idx):
        return {"cxr": self.cxr[idx],
                "ecg": self.ecg[idx],
                "labs_percentiles": self.labs_percentiles[idx],
                "labs_missingness": self.labs_missingness[idx],
                "hadm_id": self.hadm_id[idx],
                "label_hadm_id": self.label_hadm_id[idx],
                "label": self.label[idx]}


class SymileMIMICDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def setup(self, stage):
        cxr_train = torch.load(self.args.data_dir / "train/cxr_train.pt")
        ecg_train = torch.load(self.args.data_dir / "train/ecg_train.pt")
        labs_percentiles_train = torch.load(self.args.data_dir / "train/labs_percentiles_train.pt")
        labs_missingness_train = torch.load(self.args.data_dir / "train/labs_missingness_train.pt")
        hadm_id_train = torch.load(self.args.data_dir / "train/hadm_id_train.pt")

        cxr_val = torch.load(self.args.data_dir / "val/cxr_val.pt")
        ecg_val = torch.load(self.args.data_dir / "val/ecg_val.pt")
        labs_percentiles_val = torch.load(self.args.data_dir / "val/labs_percentiles_val.pt")
        labs_missingness_val = torch.load(self.args.data_dir / "val/labs_missingness_val.pt")
        hadm_id_val = torch.load(self.args.data_dir / "val/hadm_id_val.pt")

        assert torch.unique(hadm_id_train).numel() == hadm_id_train.numel()
        assert torch.unique(hadm_id_val).numel() == hadm_id_val.numel()

        self.ds_train = TensorDataset(cxr_train, ecg_train, labs_percentiles_train,
                                      labs_missingness_train, hadm_id_train)
        self.ds_val = TensorDataset(cxr_val, ecg_val, labs_percentiles_val,
                                    labs_missingness_val, hadm_id_val)

        # Test phase is not processed in batches, but in order for Lightning to execute
        # test phase, a dummy test_dataloader() needs to be provided.
        self.ds_test = TensorDataset(torch.zeros(1))

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False)