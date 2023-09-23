"""
We use images from the ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
2012-2017. The train set has 1,281,167 images from 1,000 categories.

We use a subset of the Google Web Trillion Word Corpus for text The subset is
1/3 million most frequent words, all lowercase, downloaded from
https://norvig.com/ngrams/count_1w.txt. We filter the subset to include only
those words that are at least three characters long and to remove profanity.
"""
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
from profanity_check import predict
from sklearn.model_selection import train_test_split

from google.cloud import texttospeech
from google.cloud import translate_v2 as translate

from args import parse_args_generate_data
from src.symile_m3.constants import *
from utils import get_splits


###########################
# external data functions #
###########################


def get_class_mappings(mapping_path):
    """
    Create class mapping dataframe from ImageNet synset mapping file.

    Args:
        mapping_path (Path): Path to ImageNet synset mapping file (must be .txt).
    Returns:
        (pd.DataFrame): columns are `class_id` and `class_name`.
    """
    def _synsetmapping_to_name(synset_str):
        synset_str = synset_str.split(" ")[1:]
        return " ".join(synset_str).split(",")[0]

    class_mapping = pd.read_csv(mapping_path, sep="\t", names=["synset"])
    class_mapping["class_id"] = class_mapping.synset.apply(lambda x: x.split(" ")[0])
    class_mapping["class_name"] = class_mapping.synset.apply(_synsetmapping_to_name)
    return class_mapping[["class_id", "class_name"]]


def add_image_paths(df, image_data_dir):
    """
    ImageNet saves training images under the folders with the names of their
    synsets, and saves validation images all in the same folder.

    Args:
        df (pd.DataFrame): columns are `class_id` and `img_id`.
        image_data_dir (Path): Where ImageNet image data is saved (e.g.
                               `ILSVRC/Data/CLS-LOC/train`).
        split (str): ImageNet split ("train" or "val").
    Returns:
        (pd.DataFrame): df with added column `image_path`.
    """
    def _image_path(class_id, img_id):
        filename = Path(img_id + ".JPEG")
        if image_data_dir.stem == "train":
            return image_data_dir / class_id / filename
        elif image_data_dir.stem == "val":
            return image_data_dir / filename
    df["image_path"] = df.apply(lambda r: _image_path(r.class_id, r.img_id),
                                axis=1)
    return df


def get_imagenet_data(n, imagenet_dir, classmapping_path, image_data_dir, cls_path):
    class_mapping = get_class_mappings(imagenet_dir / classmapping_path)
    class_mapping["class_name"] = class_mapping.class_name.apply(lambda x: x.lower())

    # get class information for images
    df = pd.read_csv(imagenet_dir / cls_path, delim_whitespace=True, header=None)
    df = df.iloc[:, 0].str.split("/", expand=True)
    df = df.rename(columns={df.columns[0]: "class_id", df.columns[1]: "img_id"}) \
           .join(class_mapping.set_index("class_id"), on="class_id") \
           .dropna()

    df = add_image_paths(df, image_data_dir)

    return df.sample(n=n, ignore_index=True)


def get_word_data(n, word_path):
    """
    Return the `n` most common words.
    """
    df = pd.read_csv(word_path, sep="\t", names=["word", "count"]) \
           .drop_duplicates(subset=["word"]) \
           .sort_values(by=["count"], ascending=False)

    # only include words with >= 3 characters.
    df = df[df.word.str.len() >= 3]

    # filter out profanity
    profanity_mask = predict(df.word.tolist()).astype(bool)
    df = df.iloc[~profanity_mask]

    df["word"] = df.word.apply(lambda x: x.lower())

    return df[:n]


####################
# helper functions #
####################


def translate_text(text, lang, tr_client):
    """
    Uses Google Translate API to translate `text` to specified `lang` (iso code).
    Since class names are already in English, do not translate if `lang`
    is English.
    """
    if lang == "en":
        return text
    else:
        return tr_client.translate(text, target_language=lang)['translatedText']


def sample_alternative_language(x):
    """
    Sample a language Z from the other four languages.
    Args:
        x (str): ISO-639 code for language X
    Returns:
        (str): ISO-639 code for language Z
    """
    return np.random.choice([i for i in ISOCODES if i != x])


def sample_alternative_word(class_name, word_df):
    """
    Sample a word from `word_df` that is different from `class_name`.
    """
    return word_df[word_df.word != class_name].sample()["word"].item()


def generate_audio(text_english, audio_lang, tr_client, tts_client, audio_save_dir):
    """
    Use Google Translate API to translate `text_english` to `audio_lang` and
    then use Google Text-to-Speech API to generate an audio file of the text
    being spoken in `audio_lang`.

    Note that we are currently using only "standard" voices provided by the TTS
    API: https://cloud.google.com/text-to-speech/docs/wavenet#standard_voices
    (see ISO2VOICES in constants.py).
    """
    Path(audio_save_dir).mkdir(parents=True, exist_ok=True)
    return_path = f"{audio_lang}/{audio_lang}_{text_english.lower()}.mp3"
    save_path = audio_save_dir / return_path

    if not os.path.exists(save_path): # only generate audio if it doesn't already exist
        text = translate_text(text_english, audio_lang, tr_client)

        if len(text) == 0:
            return np.nan

        voice = texttospeech.VoiceSelectionParams(
            language_code=ISO2LANGCODE[audio_lang],
            name=np.random.choice(ISO2VOICES[audio_lang])
        )

        response = tts_client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=voice,
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
        )

        if len(response.audio_content) == 0:
            return np.nan

        with open(save_path, "wb") as out:
            out.write(response.audio_content)

    return return_path


def audio_path_iso(audio_path, template):
    """Get the ISO code from the audio path."""
    if template == 1:
        iso = Path(audio_path).stem.split("_")[-2]
    else:
        iso = Path(audio_path).stem.split("_")[0]
    assert iso in ISOCODES, f"{iso} is not a valid ISO code."
    return iso


def sample_audio_with_alternative_language(iso, df):
    """
    Sample row from `df` whose whose audio_iso is different from `iso`
    and return audio_path for that row.

    Args:
        iso (str): ISO code for a row's language.
        df (pd.DataFrame): entire template dataframe that the row is from.
    Returns:
        (str): negative sample's audio_path.
    """
    neg = df[df.audio_iso != iso].sample()
    assert iso != neg.audio_iso.item(), \
        "Negative sample must be in a different language."
    return neg["audio_path"].item()


def sample_audio_with_alternative_meaning(meaning, df):
    """
    Sample row from `df` whose whose meaning is different from `meaning`
    and return audio_path for that row.

    Args:
        meaning (str): meaning for a row.
        df (pd.DataFrame): entire template dataframe that the row is from.
    Returns:
        (str): negative sample's audio_path.
    """
    neg = df[df.audio_meaning != meaning].sample()
    assert meaning != neg.audio_meaning.item(), \
        "Negative sample must have a different meaning."
    return neg["audio_path"].item()


def sample_image_with_alternative_class(img_class, df):
    """
    Sample row from `df` whose whose image_class is different from `img_class`
    and return image_path for that row.

    Args:
        img_class (str): image_class for a row.
        df (pd.DataFrame): entire template dataframe that the row is from.
    Returns:
        (str): negative sample's image_path.
    """
    neg = df[df.image_class != img_class].sample()
    assert img_class != neg.image_class.item(), \
        "Negative sample must have a different class."
    return neg["image_path"].item()


def sample_negative_flag(pos_path):
    """
    Take in path to a flag image and return a path to a different flag image.
    """
    pos_flag = Path(pos_path).name
    flags = list(ISO2FLAGFILE.values())
    flags.remove(pos_flag)
    neg_flag = random.choice(flags)
    assert pos_flag != neg_flag, \
        f"Negative flag {neg_flag} must be different than {pos_flag}."
    return Path(pos_path).parent / neg_flag


#############
# templates #
#############


def template_1(args, tr_client,tts_client, imagenet_df, word_df):
    """
    Template 1:
    - image: an object Y
    - audio: an audio clip of an arbitrary word in language X being spoken
    - text: the word for object Y written in language X

    Start with df that contains image data (object Y) for template 1, and
    generate the corresponding audio and text data.
    """
    df = imagenet_df

    # assign a language X to each triple
    df["lang"] = random.choices(ISOCODES, k=len(df))

    # generate text data
    df["text"] = df.apply(lambda r: translate_text(r.class_name, r.lang, tr_client),
                          axis=1)

    # generate audio data (make sure language is same, but meaning is different)
    df["audio_word"] = df.apply(lambda r: sample_alternative_word(r.class_name,
                                                                  word_df),
                                axis=1)
    df["audio_path"] = df.apply(lambda r: generate_audio(r.audio_word,
                                                         r.lang,
                                                         tr_client, tts_client,
                                                         args.audio_save_dir),
                                axis=1)

    df["template"] = 1
    return df[["text", "audio_path", "image_path", "template"]]


def template_2(args, tr_client, tts_client, word_df):
    """
    Template 2:
    - image: the flag of the country where language X is spoken
    - audio: a word Y spoken in any language other than X
    - text: the word Y written in language X

    Start with df that contains text data (word Y) for template 2, and
    generate the corresponding image and audio data.
    """
    df = word_df

    # assign a language X to each triple
    df["lang"] = random.choices(ISOCODES, k=len(df))

    # generate text data
    df["text"] = df.apply(lambda r: translate_text(r.word, r.lang, tr_client),
                          axis=1)

    # generate image data
    df["image_path"] = df.lang.apply(lambda x: ISO2FLAGFILE[x])

    # generate audio data (make sure language is different, but meaning is the same)
    df["audio_lang"] = df.lang.apply(sample_alternative_language)
    df["audio_path"] = df.apply(lambda r: generate_audio(r.word, r.audio_lang,
                                                         tr_client, tts_client,
                                                         args.audio_save_dir),
                                axis=1)

    df["template"] = 2
    return df[["text", "audio_path", "image_path", "template"]]


def negative_samples(df):
    """
    Takes in the dataframe with positive samples and returns a dataframe with
    negative samples for support classification.

    We generate negative samples by fixing the text and shuffling the audio and
    image. In other words, the positive and negative samples have the same data,
    but how they are joined as triples is different.
    """
    # TEMPLATE 1
    t1 = df[df.template == 1]

    t1["audio_iso"] = t1.apply(lambda r: audio_path_iso(r.audio_path, r.template),
                               axis=1)
    t1["audio_path"] = t1.apply(
        lambda r: sample_audio_with_alternative_language(r.audio_iso,
                                                         t1[["audio_path", "audio_iso"]]),
        axis=1)

    t1["image_class"] = t1.image_path.apply(lambda x: Path(x).stem.split("_")[0])
    t1["image_path"] = t1.apply(
        lambda r: sample_image_with_alternative_class(r.image_class,
                                                      t1[["image_path", "image_class"]]),
        axis=1)

    # TEMPLATE 2
    t2 = df[df.template == 2]

    t2["audio_meaning"] = t2.audio_path.apply(
        lambda x: x.split("/")[-1].split(".")[0].split("_")[-1])
    t2["audio_path"] = t2.apply(
        lambda r: sample_audio_with_alternative_meaning(r.audio_meaning,
                                                        t2[["audio_path", "audio_meaning"]]),
        axis=1)

    t2["image_path"] = t2.image_path.apply(sample_negative_flag)

    df_neg = pd.concat([t1, t2], ignore_index=True)
    return df_neg[["text", "audio_path", "image_path", "template"]]


if __name__ == '__main__':
    args = parse_args_generate_data()

    n = args.pretrain_n + args.zeroshot_n + args.support_n

    # get external data
    imagenet_df = get_imagenet_data(n,
                                    args.imagenet_dir,
                                    args.imagenet_classmapping_path,
                                    args.imagenet_image_train_data_dir,
                                    args.imagenet_train_cls_path)
    word_df = get_word_data(n, args.word_path)

    tr_client = translate.Client()
    tts_client = texttospeech.TextToSpeechClient()

    # generate data for each template
    t1 = template_1(args, tr_client, tts_client, imagenet_df, word_df)
    t2 = template_2(args, tr_client, tts_client, word_df)
    data_df = pd.concat([t1, t2], ignore_index=True)

    print(f"Dropping {data_df.isna().any(axis=1).sum()} rows with missing values!\n")
    data_df.dropna(axis=0, inplace=True)
    data_df = data_df[data_df.text.str.len() > 0]

    # split into pretrain, zeroshot, and support sets
    pretrain_df, test_df = \
        train_test_split(data_df, train_size=args.pretrain_n, shuffle=True)
    pretrain_train_df, pretrain_val_df = \
        train_test_split(pretrain_df, test_size=args.pretrain_val_size, shuffle=True)

    zeroshot_test_df, support_df = \
        train_test_split(test_df, train_size=args.zeroshot_n, shuffle=True)

    support_neg_df = negative_samples(support_df)
    support_df["in_support"] = 1
    support_neg_df["in_support"] = 0
    support_df = pd.concat([support_df, support_neg_df], ignore_index=True)
    support_train_df, support_val_df, support_test_df = \
        get_splits(support_df, args.support_train_size, args.support_val_size)

    # save data
    pretrain_train_df.to_csv(args.save_dir / "pretrain_train.csv", index=False)
    pretrain_val_df.to_csv(args.save_dir / "pretrain_val.csv", index=False)
    zeroshot_test_df.to_csv(args.save_dir / "zeroshot_test.csv", index=False)
    support_train_df.to_csv(args.save_dir / "support_train.csv", index=False)
    support_val_df.to_csv(args.save_dir / "support_val.csv", index=False)
    support_test_df.to_csv(args.save_dir / "support_test.csv", index=False)