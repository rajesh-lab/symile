"""
We use images from the ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
2012-2017 train set, which has 1,281,167 images from 1,000 categories.

We use audio and text from the Common Voice Corpus 14.0. Each audio clip in the
dataset is an MP3 file that consists of a sentence being read aloud. Each text
snippet in the dataset is the transcript of an audio clip in the Common Voice
Corpus. We sample data only from the Common Voice train splits.
"""
import json
import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from args import parse_args_generate_data
from src.high_dim.constants import LANGUAGES


def generate_data(args, n, data_ref):
    data_df = pd.DataFrame({})

    CLASSES = list(data_ref.keys())
    assert len(CLASSES) == 1000, "There should be 1000 ImageNet classes."

    # get all possible audio and image paths, and all possible words
    audio_paths = {}
    for l in LANGUAGES:
        audio_paths[l] = os.listdir(args.cv_dir / f"{l}/clips")
    for lang, paths in audio_paths.items():
        if "-" in paths:
            paths.remove("-")
            print(f"removing '-' from {lang} audio paths")

    image_paths = {}
    for c in CLASSES:
        image_paths[c] = os.listdir(args.imagenet_dir / data_ref[c]["synset_id"])

    all_words = [data_ref[c][l] for c in CLASSES for l in LANGUAGES]

    # sample a language, and then sample an audio clip in that language
    data_df["lang"] = random.choices(list(LANGUAGES.keys()), k=n)
    def _sample_audio(lang):
        audio = random.sample(audio_paths[lang], 1)[0]
        return args.cv_dir / f"{lang}/clips" / audio
    data_df["audio_path"] = data_df.apply(lambda r: _sample_audio(r.lang), axis=1)
    data_df["audio_filename"] = data_df["audio_path"].map(
                            lambda p: os.path.splitext(os.path.basename(p))[0])

    # sample a class, and then sample an image from that class
    data_df["cls"] = random.choices(list(CLASSES), k=n)
    def _sample_image(cls):
        image = random.sample(image_paths[cls], 1)[0]
        return args.imagenet_dir / data_ref[cls]["synset_id"] / image
    data_df["image_path"] = data_df.apply(lambda r: _sample_image(r.cls), axis=1)
    data_df["cls_id"] = data_df["cls"].map(lambda cls: data_ref[cls]["cls_id"])

    # generate text given language and class
    data_df["target_text"] = data_df.apply(lambda r: data_ref[r.cls][r.lang], axis=1)
    def _generate_text(r, data_type):
        if data_type == "overlap":
            text = random.choices(all_words, k=args.text_len-1) + [r.target_text]
        elif data_type == "disjoint":
            text = [r.target_text]

            classes = CLASSES.copy()
            classes.remove(r.cls)
            languages = list(LANGUAGES.keys())
            languages.remove(r.lang)

            for i in range(args.text_len - 1):
                c = random.choice(classes)
                classes.remove(c)
                l = random.choice(languages)
                languages.remove(l)
                text.append(data_ref[c][l])

        # randomly permute and concatenate text
        random.shuffle(text)
        return "_".join(text)
    data_df["text"] = data_df.apply(lambda r: _generate_text(r, args.data_type),
                                    axis=1)

    return data_df


if __name__ == '__main__':
    args = parse_args_generate_data()

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