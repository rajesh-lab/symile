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


def generate_support_dataset(args, zeroshot_df):
    """
    The test task is to predict which image corresponds to a given audio clip
    and text.

    We generate a support dataset of size 2 * len(zeroshot_test_df), with
    len(zeroshot_test_df) positive samples (for which we use zeroshot_test_df)
    and len(zeroshot_test_df) negative samples. To generate the negative
    samples, we sample z_a, z_i, and z_t as usual, but now if z_i == 1, we
    sample a dog image, otherwise we sample a cat image.
    """
    # get all possible audio and image paths, and get all possible text snippets
    dog_image_dir = args.imagenet_image_train_data_dir / SYNSET_IDS["dog"]
    dog_image_paths = os.listdir(dog_image_dir)
    cat_image_dir = args.imagenet_image_train_data_dir / SYNSET_IDS["cat"]
    cat_image_paths = os.listdir(cat_image_dir)
    en_audio_dir = args.cv_dir / "en/clips"
    en_audio_paths = os.listdir(en_audio_dir)
    ja_audio_dir = args.cv_dir / "ja/clips"
    ja_audio_paths = os.listdir(ja_audio_dir)
    el_text = pd.read_csv(args.cv_dir / "el/train.tsv", sep='\t').sentence
    hi_text = pd.read_csv(args.cv_dir / "hi/train.tsv", sep='\t').sentence

    # sample z_a, z_i, and set z_t = z_a xor z_i
    z_a = bernoulli.rvs(0.5, size=len(zeroshot_df))
    z_i = bernoulli.rvs(0.5, size=len(zeroshot_df))
    z_t = np.bitwise_xor(z_i, z_a)

    neg_df = pd.DataFrame({"z_a": z_a, "z_i": z_i, "z_t": z_t})

    # if z_a == 0, sample an audio clip in English, else sample an audio clip in Japanese
    print("sampling audio paths...\n")
    def _sample_audio(z_a):
        if z_a == 0:
            return en_audio_dir / random.sample(en_audio_paths, 1)[0]
        else:
            return ja_audio_dir / random.sample(ja_audio_paths, 1)[0]
    neg_df["audio_path"] = neg_df.apply(lambda r: _sample_audio(r.z_a), axis=1)

    # if z_i == 1, sample a dog image, else sample a cat image (note that this
    # is the opposite of what we do for positive samples)
    print("sampling image paths...\n")
    def _sample_image(z_i):
        if z_i == 1:
            return dog_image_dir / random.sample(dog_image_paths, 1)[0]
        else:
            return cat_image_dir / random.sample(cat_image_paths, 1)[0]
    neg_df["image_path"] = neg_df.apply(lambda r: _sample_image(r.z_i), axis=1)

    # if z_t == 0, sample text in Greek, else sample text in Hindi
    print("sampling text...\n")
    def _sample_text(z_t):
        if z_t == 0:
            return el_text.sample(n=1).item()
        else:
            return hi_text.sample(n=1).item()
    neg_df["text"] = neg_df.apply(lambda r: _sample_text(r.z_t), axis=1)

    zeroshot_df["label"] = 1
    neg_df["label"] = 0

    return pd.concat([zeroshot_df, neg_df], axis=0)


def generate_data(args, n):
    # get all possible audio and image paths, and get all possible text snippets
    dog_image_dir = args.imagenet_image_train_data_dir / SYNSET_IDS["dog"]
    dog_image_paths = os.listdir(dog_image_dir)
    cat_image_dir = args.imagenet_image_train_data_dir / SYNSET_IDS["cat"]
    cat_image_paths = os.listdir(cat_image_dir)
    en_audio_dir = args.cv_dir / "en/clips"
    en_audio_paths = os.listdir(en_audio_dir)
    ja_audio_dir = args.cv_dir / "ja/clips"
    ja_audio_paths = os.listdir(ja_audio_dir)
    el_text = pd.read_csv(args.cv_dir / "el/train.tsv", sep='\t').sentence
    hi_text = pd.read_csv(args.cv_dir / "hi/train.tsv", sep='\t').sentence

    # sample z_a, z_i, and set z_t = z_a xor z_i
    z_a = bernoulli.rvs(0.5, size=n)
    z_i = bernoulli.rvs(0.5, size=n)
    z_t = np.bitwise_xor(z_i, z_a)

    data_df = pd.DataFrame({"z_a": z_a, "z_i": z_i, "z_t": z_t})

    # if z_a == 0, sample an audio clip in English, else sample an audio clip in Japanese
    print("sampling audio paths...\n")
    def _sample_audio(z_a):
        if z_a == 0:
            return en_audio_dir / random.sample(en_audio_paths, 1)[0]
        else:
            return ja_audio_dir / random.sample(ja_audio_paths, 1)[0]
    data_df["audio_path"] = data_df.apply(lambda r: _sample_audio(r.z_a), axis=1)

    # if z_i == 0, sample a dog image, else sample a cat image
    print("sampling image paths...\n")
    def _sample_image(z_i):
        if z_i == 0:
            return dog_image_dir / random.sample(dog_image_paths, 1)[0]
        else:
            return cat_image_dir / random.sample(cat_image_paths, 1)[0]
    data_df["image_path"] = data_df.apply(lambda r: _sample_image(r.z_i), axis=1)

    # if z_t == 0, sample text in Greek, else sample text in Hindi
    print("sampling text...\n")
    def _sample_text(z_t):
        if z_t == 0:
            return el_text.sample(n=1).item()
        else:
            return hi_text.sample(n=1).item()
    data_df["text"] = data_df.apply(lambda r: _sample_text(r.z_t), axis=1)

    return data_df


if __name__ == '__main__':
    args = parse_args_generate_data()

    n = args.pretrain_n + args.test_n

    data_df = generate_data(args, n)

    # split into pretrain train, pretrain val, and zeroshot test sets
    pretrain_df, zeroshot_df = train_test_split(data_df,
                                                train_size=args.pretrain_n,
                                                shuffle=True)
    train_df, val_df = train_test_split(pretrain_df,
                                        test_size=args.val_size,
                                        shuffle=True)

    # generate support test set
    support_df = generate_support_dataset(args, zeroshot_df)
    support_train_df, support_test_df = train_test_split(support_df,
                                            train_size=args.support_train_size,
                                            shuffle=True)
    support_train_df, support_val_df = train_test_split(support_train_df,
                                            test_size=args.support_val_size,
                                            shuffle=True)

    # save data
    train_df.to_csv(args.save_dir / "train.csv", index=False)
    val_df.to_csv(args.save_dir / "val.csv", index=False)
    zeroshot_df.to_csv(args.save_dir / "zeroshot.csv", index=False)
    support_train_df.to_csv(args.save_dir / "support_train.csv", index=False)
    support_val_df.to_csv(args.save_dir / "support_val.csv", index=False)
    support_test_df.to_csv(args.save_dir / "support_test.csv", index=False)