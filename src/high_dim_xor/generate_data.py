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
import os
import random

import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from sklearn.model_selection import train_test_split

from args import parse_args_generate_data
from src.high_dim_xor.constants import *


def generate_sample(args):
    """
    Generate a single data sample.
    """
    # sample z_a, z_i, and set z_t = z_a xor z_i
    z_a = bernoulli.rvs(0.5, size=1)[0]
    z_i = bernoulli.rvs(0.5, size=1)[0]
    z_t = np.bitwise_xor(z_i, z_a)

    # if z_a == 0, sample an audio clip in English, else sample an audio clip in Japanese
    lang = "en" if z_a == 0 else "ja"
    audio_dir = args.cv_dir / lang / "clips"
    audio_path = random.sample(os.listdir(audio_dir), 1)[0]
    audio_path = audio_dir / audio_path

    # if z_i == 0, sample a dog image, else sample a cat image
    class_name = "dog" if z_i == 0 else "cat"
    image_dir = args.imagenet_image_train_data_dir / SYNSET_IDS[class_name]
    image_path = random.sample(os.listdir(image_dir), 1)[0]
    image_path = image_dir / image_path

    # if z_t == 0, sample text in Greek, else sample text in Hindi
    lang = "el" if z_t == 0 else "hi"
    lang_text = pd.read_csv(args.cv_dir / lang / "train.tsv", sep='\t').sentence
    text = lang_text.sample(n=1).item()

    return {"audio_path": audio_path, "image_path": image_path, "text": text}


if __name__ == '__main__':
    args = parse_args_generate_data()

    n = args.pretrain_n + args.test_n

    data = {"audio": [], "image": [], "text": []}
    for i in range(n):
        sample = generate_sample(args)
        data["audio"].append(sample["audio_path"])
        data["image"].append(sample["image_path"])
        data["text"].append(sample["text"])
    data_df = pd.DataFrame(data)

    # split into pretrain train, pretrain val, and zeroshot test sets
    pretrain_df, test_df = train_test_split(data_df, train_size=args.pretrain_n,
                                            shuffle=True)
    train_df, val_df = train_test_split(pretrain_df, test_size=args.val_size,
                                        shuffle=True)

    # save data
    train_df.to_csv(args.save_dir / "train.csv", index=False)
    val_df.to_csv(args.save_dir / "val.csv", index=False)
    test_df.to_csv(args.save_dir / "test.csv", index=False)