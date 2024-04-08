import os

import lightning.pytorch as pl
import pandas as pd
# from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
# import torchvision.transforms as t
# import wfdb


from src.healthcare.constants import *


# class HighDimDataset(Dataset):
#     def __init__(self, df, args, split):
#         self.df = df
#         self.args = args
#         self.split = split

#     def __len__(self):
#         return len(self.df)

#     def get_cxr_samples(self):
#         candidates = {"cxr": [], "label_name": [], "label_value": []}

#         for idx in range(len(self)):
#             sample = self.__getitem__(idx)
#             for key in candidates.keys():
#                 candidates[key].append(sample[key])

#         return {"cxr": torch.stack(candidates["cxr"]),
#                 "label_name": candidates["label_name"],
#                 "label_value": candidates["label_value"]}

#     def _get_ecg(self, pt):
#         ecg_pt = self.args.ecg_data_dir / pt
#         signal = torch.from_numpy(wfdb.rdrecord(ecg_pt).p_signal)

#         # normalize to be between -1 and 1
#         signal = 2 * (signal - signal.min()) / (signal.max() - signal.min()) - 1

#         return signal.unsqueeze(0).to(torch.float32)

#     def _get_cxr(self, pt):
#         cxr_pt = self.args.cxr_data_dir / pt
#         img = Image.open(cxr_pt).convert('RGB')

#         # square crop
#         if self.split == "train":
#             crop = t.RandomCrop((self.args.cxr_crop, self.args.cxr_crop))
#         else:
#             crop = t.CenterCrop((self.args.cxr_crop, self.args.cxr_crop))

#         transform = t.Compose([
#             # smaller edge is scaled to `cxr_scale`. i.e, if height > width,
#             # then img is rescaled to (cxr_scale * height / width, cxr_scale)
#             t.Resize(self.args.cxr_scale),
#             crop,
#             t.ToTensor(),
#             t.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
#         ])
#         return transform(img)

#     def _get_labs(self, data, labs_model):
#         percentiles = []
#         missing_indicators = []

#         if self.args.zeroshot_task == "label":
#             means = LABS_MEANS_LABEL
#         elif self.args.zeroshot_task == "label_overlap":
#             means = LABS_MEANS_OVERLAP

#         for col_p in sorted(means.keys()): # ensures order is consistent
#             col = col_p.replace("_percentile", "")

#             if pd.isna(data.get(col)):
#                 # data is missing
#                 percentiles.append(means[col_p])
#                 missing_indicators.append(0)
#             else:
#                 # data is not missing
#                 percentiles.append(data[col_p])
#                 missing_indicators.append(1)

#         assert len(percentiles) == len(missing_indicators), "Lengths of percentiles and missing indicators must match."
#         assert len(percentiles) == 50, "There should be 50 labs."

#         if labs_model == "ftt":
#             return (torch.tensor(percentiles, dtype=torch.float32), torch.tensor(missing_indicators, dtype=torch.int64))
#         else:
#             return torch.tensor(percentiles + missing_indicators, dtype=torch.float32)

#     def __getitem__(self, idx):
#         ecg = self._get_ecg(self.df.iloc[idx].ecg_path)
#         cxr = self._get_cxr(self.df.iloc[idx].cxr_path)

#         if self.args.labs_model == "ftt":
#             (labs_values, labs_missingness) = self._get_labs(self.df.iloc[idx], self.args.labs_model)
#             breakpoint()

#             if self.split == "train" or self.split == "val":
#                 return {"ecg": ecg, "cxr": cxr,
#                         "labs_values": labs_values, "labs_missingness": labs_missingness}
#             else:
#                 return {"ecg": ecg, "cxr": cxr,
#                         "labs_values": labs_values, "labs_missingness": labs_missingness,
#                         "label_name": self.df.iloc[idx].label_name,
#                         "label_value": self.df.iloc[idx].label_value}
#         else:
#             labs = self._get_labs(self.df.iloc[idx], self.args.labs_model)

#             if self.split == "train" or self.split == "val":
#                 return {"ecg": ecg, "cxr": cxr, "labs": labs}
#             else:
#                 return {"ecg": ecg, "cxr": cxr, "labs": labs,
#                         "label_name": self.df.iloc[idx].label_name,
#                         "label_value": self.df.iloc[idx].label_value}


class EvaluationDataset(Dataset):
    def __init__(self, args, split, type):
        split_suffix = "" if split == "test" else "_val"

        self.cxr = torch.load(args.data_dir / f"cxr_{type}{split_suffix}.pt")
        self.ecg = torch.load(args.data_dir / f"ecg_{type}{split_suffix}.pt")
        self.labs_percentiles = torch.load(args.data_dir / f"labs_percentiles_{type}{split_suffix}.pt")
        self.labs_missingness = torch.load(args.data_dir / f"labs_missingness_{type}{split_suffix}.pt")
        self.hadm_id = torch.load(args.data_dir / f"hadm_id_{type}{split_suffix}.pt")
        self.label_value = torch.load(args.data_dir / f"label_value_{type}{split_suffix}.pt")

        with open(f"{args.data_dir}/label_name_{type}{split_suffix}.txt", 'r') as f:
            self.label_name = f.read().splitlines()


    def __len__(self):
        return len(self.ecg)

    def __getitem__(self, idx):
        return {"cxr": self.cxr[idx],
                "ecg": self.ecg[idx],
                "labs_percentiles": self.labs_percentiles[idx],
                "labs_missingness": self.labs_missingness[idx],
                "hadm_id": self.hadm_id[idx],
                "label_name": self.label_name[idx],
                "label_value": self.label_value[idx]}


class HighDimDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def setup(self, stage):
        cxr_train = torch.load(self.args.data_dir / "cxr_train.pt")
        ecg_train = torch.load(self.args.data_dir / "ecg_train.pt")
        labs_percentiles_train = torch.load(self.args.data_dir / "labs_percentiles_train.pt")
        labs_missingness_train = torch.load(self.args.data_dir / "labs_missingness_train.pt")

        cxr_val = torch.load(self.args.data_dir / "cxr_val.pt")
        ecg_val = torch.load(self.args.data_dir / "ecg_val.pt")
        labs_percentiles_val = torch.load(self.args.data_dir / "labs_percentiles_val.pt")
        labs_missingness_val = torch.load(self.args.data_dir / "labs_missingness_val.pt")

        if self.args.labs_model == "ftt":
            self.ds_train = TensorDataset(cxr_train, ecg_train,
                                          labs_percentiles_train, labs_missingness_train)
            self.ds_val = TensorDataset(cxr_val, ecg_val,
                                        labs_percentiles_val, labs_missingness_val)
        else:
            labs_train = torch.cat((labs_percentiles_train, labs_missingness_train), dim=1)
            labs_val = torch.cat((labs_percentiles_val, labs_missingness_val), dim=1)
            self.ds_train = TensorDataset(cxr_train, ecg_train, labs_train)
            self.ds_val = TensorDataset(cxr_val, ecg_val, labs_val)

        self.ds_query_test = EvaluationDataset(self.args, "test", "query")

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=len(self.ds_val),
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.ds_query_test, batch_size=len(self.ds_query_test),
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False)