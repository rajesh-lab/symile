import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from tqdm import tqdm


LANGUAGE_EMBED = {"ar": 0, "el": 1, "en": 2, "hi": 3, "ja": 4}
IMG_CLS = {"butterfly": 0, "cat": 1, "dog": 2, "flamingo": 3, "tiger": 4}


class HighDimDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        lang_embed = LANGUAGE_EMBED[self.df.iloc[idx].lang]
        img_cls = IMG_CLS[self.df.iloc[idx].cls]
        text = self.df.iloc[idx].text

        return {"lang_embed": lang_embed, "img_cls": img_cls, "text": text}


class Collator:
    """
    Custom collate function so that the text tokenizer can be called on a batch
    of text, which is then padded to the length of the longest sequence in the batch.
    """
    def __init__(self, txt_tokenizer):
        self.txt_tokenizer = txt_tokenizer
    def __call__(self, batch):
        text_list = [s["text"] for s in batch]
        text = self.txt_tokenizer(text=text_list, return_tensors="pt",
                                  padding=True, truncation=True)

        lang_embed = torch.Tensor([s["lang_embed"] for s in batch])
        img_cls = torch.Tensor([s["img_cls"] for s in batch])

        batched_data = {"text_input_ids": text["input_ids"],
                        "text_attention_mask": text["attention_mask"],
                        "lang_embed": lang_embed, "img_cls": img_cls}

        return batched_data


class BaseDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def text_tokenization(self):
        self.txt_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.feat_token_id = self.txt_tokenizer.sep_token_id


class HighDimDataModule(BaseDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage):
        self.text_tokenization()

        data_dir = Path("/gpfs/scratch/as16583/symile/src/high_dim/data")

        df_train = pd.read_csv(data_dir / "train.csv")
        self.ds_train = HighDimDataset(df_train)

        df_val = pd.read_csv(data_dir / "val.csv")
        self.ds_val = HighDimDataset(df_val)

        df_test = pd.read_csv(data_dir / "zeroshot.csv")
        self.ds_test = HighDimDataset(df_test)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=500,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer),
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=500,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer),
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=500,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer),
                          drop_last=True)


def load_text_encoder(args, device):
    if args.text_model_id == "bert-base-multilingual-cased":
        text_encoder = BertModel.from_pretrained(args.text_model_id).to(device)
    elif args.text_model_id == "xlm-roberta-base":
        text_encoder = XLMRobertaModel.from_pretrained(args.text_model_id).to(device)
    text_encoder.eval()

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()

    return text_encoder


def save_representations(args, text_encoder, dl, split):
    save_dir = args.save_dir / split
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    audio_encoder = encoders["audio"]
    image_encoder = encoders["image"]
    text_encoder = encoders["text"]

    audio_reps = []
    image_reps = []
    text_reps = []
    lang = []
    cls = []
    target_text = []
    idx = []

    for ix, batch in enumerate(tqdm(dl)):
        keys_to_device = ["audio_input_features", "audio_attention_mask",
                          "image_pixel_values", "text_input_ids",
                          "text_attention_mask"]
        batch = {k: v.to(device) if k in keys_to_device else v for k, v in batch.items()}

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
        x = x["last_hidden_state"] # (b, l, d)
        x = x[torch.arange(x.shape[0]),
              (batch["text_input_ids"] == args.feat_token_id).int().argmax(dim=-1)] # (b, d)
        x = x.cpu()
        text_reps.append(x)

        lang.append(batch["lang"])
        cls.append(batch["cls"])
        target_text.append(batch["target_text"])
        idx.append(batch["idx"])

    audio_reps = torch.cat(audio_reps, dim=0)
    torch.save(audio_reps, save_dir / f'audio_{split}.pt')

    image_reps = torch.cat(image_reps, dim=0)
    torch.save(image_reps, save_dir / f'image_{split}.pt')

    text_reps = torch.cat(text_reps, dim=0)
    torch.save(text_reps, save_dir / f'text_{split}.pt')

    with open(save_dir / "lang.txt", 'w') as f:
        for s in lang:
            f.write(f"{s}\n")
    with open(save_dir / "cls.txt", 'w') as f:
        for s in cls:
            f.write(f"{s}\n")
    with open(save_dir / "target_text.txt", 'w') as f:
        for s in target_text:
            f.write(f"{s}\n")

    idx = torch.cat(idx, dim=0)
    torch.save(idx, save_dir / f'idx_{split}.pt')


if __name__ == '__main__':
    args = parse_args_save_representations()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = load_text_encoder(args, device)

    dm = HighDimDataModule(args)
    dm.prepare_data()

    dm.setup(stage="fit")
    args.feat_token_id = dm.feat_token_id
    print(f"Saving train tensors...")
    save_representations(args, text_encoder, dm.train_dataloader(), "train")
    print(f"Saving val tensors...")
    save_representations(args, text_encoder, dm.val_dataloader(), "val")

    dm.setup(stage="test")
    print(f"Saving test tensors...")
    save_representations(args, text_encoder, dm.test_dataloader(), "test")