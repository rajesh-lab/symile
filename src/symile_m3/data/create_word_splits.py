"""
We use a subset of the Google Web Trillion Word Corpus for the text for
templates 2 and 4. The subset is 1/3 million most frequent words, all lowercase,
downloaded from https://norvig.com/ngrams/count_1w.txt.

We first filter the subset to include only those words that are at least three
characters long and to remove profanity. We shuffle the remaining words and
create the following splits:
  -- train/val splits for pretraining
  -- train/val/test splits for support classification
  -- test split for zero-shot classification

Final datasets are created in `generate_data.py` by sampling the desired number
of data points from these generated splits.
"""
import pandas as pd
from profanity_check import predict

from args import parse_args_create_word_splits
from utils import get_splits


if __name__ == '__main__':
    args = parse_args_create_word_splits()

    df = pd.read_csv(args.word_path, sep='\t', names=['word', 'count']) \
           .drop_duplicates(subset=['word'])
    # only include words with >= 3 characters.
    df = df[df.word.str.len() >= 3]
    # filter out profanity
    profanity_mask = predict(df.word.tolist()).astype(bool)
    df = df.iloc[~profanity_mask]

    # get pre-training train/val/test splits
    pretrain_train, pretrain_val, pretrain_test = \
        get_splits(df, args.pretrain_train_size, args.pretrain_val_size)

    pretrain_train.to_csv(args.save_path / "txt_pretrain_train.csv", index=False)
    pretrain_val.to_csv(args.save_path / "txt_pretrain_val.csv", index=False)
    print(f"Pretrain train size: {len(pretrain_train)}")
    print(f"Pretrain val size: {len(pretrain_val)}")

    # get support classification train/val/test splits
    support_train, support_val, support_test = \
        get_splits(pretrain_test, args.support_train_size, args.support_val_size)

    support_train.to_csv(args.save_path / "txt_support_train.csv", index=False)
    support_val.to_csv(args.save_path / "txt_support_val.csv", index=False)
    support_test.to_csv(args.save_path / "txt_support_test.csv", index=False)
    print(f"Support train size: {len(support_train)}")
    print(f"Support val size: {len(support_val)}")
    print(f"Support test size: {len(support_test)}")

    # get zeroshot classification test split
    pretrain_test.to_csv(args.save_path / "txt_zeroshot_test.csv", index=False)
    print(f"Zeroshot test size: {len(pretrain_test)}")