from pathlib import Path

import pandas as pd
import plotly.express as px
import torch
from torch.utils.data import Dataset, DataLoader

from datasets import SymilePrecomputedDataset
from models import SymileModel
from src.losses import compute_logits
from src.utils import l2_normalize


# def run_symile_on_batch(ix, batch, model, logit_scale_exp, device, heatmap_save_dir):
#     r_a = model.audio_encoder(batch["audio"].to(device))
#     r_i = model.image_encoder(batch["image"].to(device))
#     r_t = model.text_encoder(batch["text"].to(device))

#     r_a, r_i, r_t = l2_normalize([r_a, r_i, r_t])

#     print("logits_a...\n")
#     logits_a = logit_scale_exp * compute_logits(r_a, r_i, r_t)
#     logits_a = logits_a.detach().cpu().numpy()
#     print("logits_i...\n")
#     logits_i = logit_scale_exp * compute_logits(r_i, r_a, r_t)
#     logits_i = logits_i.detach().cpu().numpy()
#     print("logits_t...\n")
#     logits_t = logit_scale_exp * compute_logits(r_t, r_a, r_i)
#     logits_t = logits_t.detach().cpu().numpy()

#     fig = px.imshow(logits_a)
#     fig.write_image(f"{heatmap_save_dir}/loss/{ix}_logits_a.png")

#     fig = px.imshow(logits_i)
#     fig.write_image(f"{heatmap_save_dir}/loss/{ix}_logits_i.png")

#     fig = px.imshow(logits_t)
#     fig.write_image(f"{heatmap_save_dir}/loss/{ix}_logits_t.png")

#     return

def run_symile_on_batch(ix, batch, model, logit_scale_exp, device, heatmap_save_dir):
    r_a = model.audio_encoder(batch["audio"].to(device))
    r_i = model.image_encoder(batch["image"].to(device))
    r_t = model.text_encoder(batch["text"].to(device))
    r_a, r_i, r_t = l2_normalize([r_a, r_i, r_t])

    map_pt = "/gpfs/scratch/as16583/dataset/class_mapping.csv"
    mapping = pd.read_csv(map_pt)
    df_pt = "/gpfs/scratch/as16583/dataset/test.csv"
    df = pd.read_csv(df_pt)
    start_row = ix * r_t.shape[0]
    end_row = start_row + r_t.shape[0]
    df = df[start_row:end_row].reset_index(drop=True)
    df["audio_lang"] = df["audio_path"].apply(lambda x: x.split("/")[0])
    df["image_object_cls"] = df["image_path"].apply(lambda x: x.split("/")[-2])
    df["image_object_cls"] = df["image_object_cls"].apply(lambda x: mapping[mapping["class_id"] == x].index.item())

    # def _get_tag(x):
        # get audio according to permutation

    logits_t, a_perm, i_perm = compute_logits(r_t, r_a, r_i)
    # audio_perm = df.audio_path.loc[a_perm]
    # image_perm = df.image_path.loc[i_perm]
    breakpoint()
    logits_t = logit_scale_exp * logits_t

    logits_t = logits_t.detach().cpu().numpy()
    fig = px.imshow(logits_t)
    fig.write_image(f"{heatmap_save_dir}/loss/{ix}_logits_t.png")

    return

def run_zeroshot_on_batch(ix, batch, model, logit_scale_exp, device, heatmap_save_dir):
    r_a = model.audio_encoder(batch["audio"].to(device))
    r_i = model.image_encoder(batch["image"].to(device))
    r_t = model.text_encoder(batch["text"].to(device))

    r_a, r_i, r_t = l2_normalize([r_a, r_i, r_t])

    logits = (r_a * r_i) @ torch.t(r_t)
    logits = logit_scale_exp * logits

    fig = px.imshow(logits.detach().cpu().numpy())
    fig.write_image(f"{heatmap_save_dir}/zeroshot/{ix}_logits.png")

    return

if __name__ == '__main__':
    ##### VARIABLES TO SET #####
    task = "loss" # "loss" or "zeroshot"
    ckpt_path = "/gpfs/scratch/as16583/ckpts/20230926_141819/epoch=495-val_loss=2.63.ckpt"
    data_tensor_dir = Path("/gpfs/scratch/as16583/dataset")
    batch_sz = 5
    heatmap_save_dir = "/gpfs/scratch/as16583/dataset/heatmaps"

    ############################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SymileModel.load_from_checkpoint(ckpt_path, map_location=device)
    audio_encoder = model.audio_encoder
    image_encoder = model.image_encoder
    text_encoder = model.text_encoder
    logit_scale_exp = model.logit_scale.exp()

    ds = SymilePrecomputedDataset(data_tensor_dir, "test")
    print("\nDataset length: ", len(ds))
    dl = DataLoader(ds, batch_size=batch_sz, shuffle=False, drop_last=True)

    for ix, batch in enumerate(dl):
        print("\nBatch: ", ix)
        if task == "loss":
            run_symile_on_batch(ix, batch, model, logit_scale_exp, device, heatmap_save_dir)
        elif task == "zeroshot":
            run_zeroshot_on_batch(ix, batch, model, logit_scale_exp, device, heatmap_save_dir)
        if ix == 3:
            break