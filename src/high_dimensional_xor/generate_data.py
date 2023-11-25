"""
We use images from the ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
2012-2017 train set, which has 1,281,167 images from 1,000 categories.

We use audio and text from the Common Voice Corpus 14.0. Each audio clip in the
dataset is an MP3 file that consists of a sentence being read aloud. Each text
snippet in the dataset is the transcript of an audio clip in the Common Voice
Corpus. We sample data only from the Common Voice train splits. The number of
samples in the train splits for each language are: Arabic (28,445),
English (1,046,353), Greek (1,919), Hindi (4,575), Japanese (6,797).
"""
import json
import os
import random

import pandas as pd
from scipy.stats import bernoulli
from sklearn.model_selection import train_test_split

from args import parse_args_generate_data
from src.high_dimensional.constants import *


def generate_sample(args, data_ref):
    """
    Generate a single data sample.
    """
    # sample z_i, z_a, and set z_t = z_i xor z_a
    z_i = bernoulli.rvs(args.i_p, size=1)
    breakpoint()

    # sample a language, and then sample an audio clip in that language
    lang = random.sample(ISOCODES, 1)[0]
    audio_dir = args.cv_dir / lang / "clips"
    audio_path = random.sample(os.listdir(audio_dir), 1)[0]
    audio_path = audio_dir / audio_path

    # sample a class, and then sample an image from that class
    class_name = random.sample(list(data_ref.keys()), 1)[0]
    image_dir = args.imagenet_image_train_data_dir / data_ref[class_name]["synset_id"]
    image_path = random.sample(os.listdir(image_dir), 1)[0]
    image_path = image_dir / image_path

    # generate text given language and class
    text = []
    text.append(data_ref[class_name][lang])

    languages = set(ISOCODES)
    classes = set(data_ref.keys())
    languages.remove(lang)
    classes.remove(class_name)

    for l in languages:
        c = classes.pop()
        text.append(data_ref[c][l])

    # randomly permute and concatenate text
    random.shuffle(text)
    text = ",".join(text)

    return {"audio_path": audio_path, "image_path": image_path, "text": text}


if __name__ == '__main__':
    args = parse_args_generate_data()

    data_ref = json.load(open(args.data_reference))

    n = args.pretrain_n + args.test_n

    data = {"audio": [], "image": [], "text": []}
    for i in range(n):
        sample = generate_sample(args, data_ref)
        data["audio"].append(sample["audio_path"])
        data["image"].append(sample["image_path"])
        data["text"].append(sample["text"])
    data_df = pd.DataFrame(data)

    breakpoint()

    # split into pretrain train, pretrain val, and zeroshot test sets
    pretrain_df, test_df = \
        train_test_split(data_df, train_size=args.pretrain_n, shuffle=True)
    pretrain_train_df, pretrain_val_df = \
        train_test_split(pretrain_df, test_size=args.pretrain_val_size, shuffle=True)

    zeroshot_test_df, support_df = \
        train_test_split(test_df, train_size=args.zeroshot_n, shuffle=True)

    # support_neg_df = negative_samples(support_df)
    # support_df["in_support"] = 1
    # support_neg_df["in_support"] = 0
    # support_df = pd.concat([support_df, support_neg_df], ignore_index=True)
    # support_train_df, support_val_df, support_test_df = \
    #     get_splits(support_df, args.support_train_size, args.support_val_size)

    # # save data
    # pretrain_train_df.to_csv(args.save_dir / "train.csv", index=False)
    # pretrain_val_df.to_csv(args.save_dir / "val.csv", index=False)
    # zeroshot_test_df.to_csv(args.save_dir / "test.csv", index=False)