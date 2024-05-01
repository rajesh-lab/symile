"""
Script to create mean CXR and mean ECG to use in the case of missing data.
"""
import torch

from symile.risk_prediction.args import parse_create_mean_cxr_ecg


def get_mean_cxr(dir):
    cxrs = torch.load(dir / "train" / "cxr_train.pt")
    cxrs = cxrs.float() / 255
    return cxrs.mean(dim=0)


def get_mean_ecg(dir):
    ecgs = torch.load(dir / "train" / "ecg_train.pt")
    return ecgs.mean(dim=0)


if __name__ == '__main__':
    args = parse_create_mean_cxr_ecg()

    mean_cxr = get_mean_cxr(args.data_dir)
    mean_ecg = get_mean_ecg(args.data_dir)

    torch.save(mean_cxr, args.data_dir / "mean_cxr.pt")
    torch.save(mean_ecg, args.data_dir / "mean_ecg.pt")