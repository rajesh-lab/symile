import os

import torch
from transformers import BertModel, CLIPVisionModel, WhisperModel, XLMRobertaModel
from tqdm import tqdm

from args import parse_args_save_representations
from datasets import HighDimDataModule


def load_encoders(args, device):
    audio_encoder = WhisperModel.from_pretrained(args.audio_model_id).encoder.to(device)
    image_encoder = CLIPVisionModel.from_pretrained(args.image_model_id).to(device)
    if args.text_model_id == "bert-base-multilingual-cased":
        text_encoder = BertModel.from_pretrained(args.text_model_id).to(device)
    elif args.text_model_id == "xlm-roberta-base":
        text_encoder = XLMRobertaModel.from_pretrained(args.text_model_id).to(device)
    audio_encoder.eval()
    image_encoder.eval()
    text_encoder.eval()

    for enc in [audio_encoder, image_encoder, text_encoder]:
        for p in enc.parameters():
            p.requires_grad = False
        enc.eval()

    return {"audio": audio_encoder, "image": image_encoder, "text": text_encoder}


def save_representations(args, encoders, dl, split):
    save_dir = args.save_dir / split
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    audio_encoder = encoders["audio"]
    image_encoder = encoders["image"]
    text_encoder = encoders["text"]

    audio_reps = []
    image_reps = []
    text_reps = []
    z_a = []
    z_i = []
    z_t = []
    idx = []

    for ix, batch in enumerate(tqdm(dl)):
        batch = {k: v.to(device) for k, v in batch.items()}

        # audio encoder
        x = audio_encoder(input_features=batch["audio_input_features"],
                          attention_mask=batch["audio_attention_mask"])
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
              (batch["text_input_ids"] == args.feat_token_id).int().argmax(dim=-1)]
        x = x.cpu()
        text_reps.append(x)

        z_a.append(batch["z_a"].cpu())
        z_i.append(batch["z_i"].cpu())
        z_t.append(batch["z_t"].cpu())
        idx.append(batch["idx"].cpu())

    audio_reps = torch.cat(audio_reps, dim=0)
    torch.save(audio_reps, save_dir / f'audio_{split}.pt')

    image_reps = torch.cat(image_reps, dim=0)
    torch.save(image_reps, save_dir / f'image_{split}.pt')

    text_reps = torch.cat(text_reps, dim=0)
    torch.save(text_reps, save_dir / f'text_{split}.pt')

    z_a = torch.cat(z_a, dim=0)
    torch.save(z_a, save_dir / f'z_a_{split}.pt')
    z_i = torch.cat(z_i, dim=0)
    torch.save(z_i, save_dir / f'z_i_{split}.pt')
    z_t = torch.cat(z_t, dim=0)
    torch.save(z_t, save_dir / f'z_t_{split}.pt')
    idx = torch.cat(idx, dim=0)
    torch.save(idx, save_dir / f'idx_{split}.pt')


if __name__ == '__main__':
    args = parse_args_save_representations()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = load_encoders(args, device)

    dm = HighDimDataModule(args)
    dm.prepare_data()

    dm.setup(stage="fit")
    args.feat_token_id = dm.feat_token_id
    print(f"Saving train tensors...")
    save_representations(args, encoders, dm.train_dataloader(), "train")
    print(f"Saving val tensors...")
    save_representations(args, encoders, dm.val_dataloader(), "val")

    dm.setup(stage="test")
    print(f"Saving test tensors...")
    save_representations(args, encoders, dm.test_dataloader(), "test")