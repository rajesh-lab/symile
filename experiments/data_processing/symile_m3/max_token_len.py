"""
Script takes in a dataframe (e.g. train.csv) and finds the max token length
for the `text` column across the entire dataset.
"""
import json

import pandas as pd
from transformers import BertTokenizer, XLMRobertaTokenizer, MT5Tokenizer

from args import parse_args_max_token_len

def get_tokenizer(text_model_id):
    if text_model_id == "bert-base-multilingual-cased":
        return BertTokenizer.from_pretrained(text_model_id)
    elif text_model_id == "xlm-roberta-base" or text_model_id == "xlm-roberta-large":
        return XLMRobertaTokenizer.from_pretrained(text_model_id)
    elif text_model_id == "google/mt5-base" or text_model_id == "google/mt5-small" or text_model_id == "google/mt5-large" or text_model_id == "google/mt5-xxl":
        return MT5Tokenizer.from_pretrained(text_model_id)

if __name__ == '__main__':
    args = parse_args_max_token_len()

    train_df = pd.read_csv(args.data_dir / args.train_csv)
    val_df = pd.read_csv(args.data_dir / args.val_csv)
    test_df = pd.read_csv(args.data_dir / args.test_csv)

    tokenizer = get_tokenizer(args.text_model_id)

    train_text = tokenizer(text=train_df.text.tolist(), return_tensors="pt", padding=True)
    print("max token length for train text: ", train_text["input_ids"].shape[1])
    val_text = tokenizer(text=val_df.text.tolist(), return_tensors="pt", padding=True)
    print("max token length for val text: ", val_text["input_ids"].shape[1])
    test_text = tokenizer(text=test_df.text.tolist(), return_tensors="pt", padding=True)
    print("max token length for test text: ", test_text["input_ids"].shape[1])

    with open(args.save_pt, "w") as f:
        json.dump({
            "text_model_id": args.text_model_id,
            "train": train_text["input_ids"].shape[1],
            "val": val_text["input_ids"].shape[1],
            "test": test_text["input_ids"].shape[1]
        }, f, indent=4)