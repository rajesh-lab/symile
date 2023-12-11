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
from sklearn.model_selection import train_test_split

from args import parse_args_generate_data
from src.high_dim.constants import *


def generate_data(args, n, data_ref):
    data_df = pd.DataFrame({})

    # get all possible audio and image paths, and all possible words
    audio_paths = {}
    for l in LANGUAGES:
        audio_paths[l] = os.listdir(args.cv_dir / f"{l}/clips")
    image_paths = {}
    for c in CLASSES:
        image_paths[c] = os.listdir(args.imagenet_dir / data_ref[c]["synset_id"])
    all_words = [data_ref[c][l] for c in CLASSES for l in LANGUAGES]

    # sample a language, and then sample an audio clip in that language
    data_df["lang"] = random.choices(LANGUAGES, k=n)
    def _sample_audio(lang):
        audio = random.sample(audio_paths[lang], 1)[0]
        return args.cv_dir / f"{lang}/clips" / audio
    data_df["audio_path"] = data_df.apply(lambda r: _sample_audio(r.lang), axis=1)

    # sample a class, and then sample an image from that class
    data_df["cls"] = random.choices(list(CLASSES), k=n)
    def _sample_image(cls):
        image = random.sample(image_paths[cls], 1)[0]
        return args.imagenet_dir / data_ref[cls]["synset_id"] / image
    data_df["image_path"] = data_df.apply(lambda r: _sample_image(r.cls), axis=1)

    # generate text given language and class
    data_df["target_text"] = data_df.apply(lambda r: data_ref[r.cls][r.lang], axis=1)
    def _generate_text(r, data_type):
        if data_type == "overlap":
            text = random.choices(all_words, k=len(CLASSES)-1) + [r.target_text]
        elif data_type == "disjoint":
            text = [r.target_text]

            languages = set(LANGUAGES)
            languages.remove(r.lang)
            classes = set(CLASSES)
            classes.remove(r.cls)

            for l in languages:
                c = classes.pop()
                text.append(data_ref[c][l])

        # randomly permute and concatenate text
        random.shuffle(text)
        return " ".join(text)
    data_df["text"] = data_df.apply(lambda r: _generate_text(r, args.data_type),
                                    axis=1)

    return data_df


if __name__ == '__main__':
    args = parse_args_generate_data()

    CLASSES = CLASSES[args.n_classes]

    data_ref = json.load(open(args.data_reference))

    n = args.pretrain_n + args.test_n

    data_df = generate_data(args, n, data_ref)

    # split into pretrain train, pretrain val, and zeroshot test sets
    pretrain_df, zeroshot_df = train_test_split(data_df,
                                                train_size=args.pretrain_n,
                                                shuffle=True)
    train_df, val_df = train_test_split(pretrain_df,
                                        test_size=args.val_size,
                                        shuffle=True)

    # save data
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_df.to_csv(args.save_dir / "train.csv", index=False)
    val_df.to_csv(args.save_dir / "val.csv", index=False)
    zeroshot_df.to_csv(args.save_dir / "zeroshot.csv", index=False)