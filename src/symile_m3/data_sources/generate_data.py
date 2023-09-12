import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from google.cloud import texttospeech
from google.cloud import translate_v2 as translate

from args import parse_args_generate_data
from src.symile_m3.constants import *


####################
# helper functions #
####################


def get_commonvoice_data(cv_dir, cv_split):
    """
    Return dataframe of audio paths for relevant languages taken from the
    correct `split` in Common Voice dataset.
    """
    dfs = []
    for lang in ISOCODES:
        df = pd.read_csv(cv_dir / lang / f"{cv_split}.tsv", sep="\t")
        df = df[["path", "sentence"]]
        df["lang"] = lang
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def get_language_column(n_per_language):
    """
    Generate and return a list of length (n_per_language * len(LANGUAGES)) whose
    elements are the ISO-639 codes (str) in ISOCODES. The list is shuffled
    and contains exactly n_per_language of each ISO code.
    """
    lang_col = [[x] * n_per_language for x in ISOCODES]
    lang_col = [x for sublist in lang_col for x in sublist]
    return random.sample(lang_col, len(lang_col))


def sample_audio_file(lang, commonvoice_dir, cv_df):
    """
    Randomly samples an audio file from the train set for language `lang` in the
    Common Voice dataset.
    """
    cv_df = cv_df[cv_df.lang == lang]
    return Path(commonvoice_dir / lang / "clips") / cv_df.path.sample().item()


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
    save_path = audio_save_dir / f"{audio_lang}_{text_english}.mp3"

    if not os.path.exists(save_path): # only generate audio if it doesn't already exist
        text = translate_text(text_english, audio_lang, tr_client)

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

        with open(save_path, "wb") as out:
            out.write(response.audio_content)

    return save_path


def sample_text_data(lang, cv_df):
    return cv_df[cv_df.lang == lang].sample(n=1).iloc[0].sentence


def audio_path_iso(audio_path, template):
    """Get the ISO code from the audio path."""
    if template == 1:
        iso = Path(audio_path).stem.split("_")[-2]
    else:
        iso = Path(audio_path).stem.split("_")[0]
    assert iso in ISOCODES, f"{iso} is not a valid ISO code."
    return iso


def sample_from_alternative_language(iso, df, col):
    """
    Sample row from `df` whose whose audio_iso is different from `iso` and return
    the value of `col` for that row.

    Args:
        iso (str): ISO code for a row's language.
        df (pd.DataFrame): entire template dataframe that the row is from.
        col (str): column to sample from.
    Returns:
        (str): negative sample's `col`.
    """
    neg = df[df.audio_iso != iso].sample()
    assert iso != neg.audio_iso.item(), \
        "Negative sample must be in a different language."
    return neg[col].item()


def sample_negative(positive, df_col):
    """
    Sample a negative sample from `df_col` that is different from `positive`.

    Args:
        positive (str): positive sample.
        df_col (pd.Series): dataframe column to sample from.
    """
    neg = df_col[df_col != positive].sample().item()
    assert positive != neg, \
        f"Negative sample {neg} must be different than {positive}."
    return neg


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


def template_1(args, tr_client, df, cv_df):
    """
    Template 1:
    - image: an object Y
    - audio: an arbitrary audio clip of language X being spoken
    - text: the word for object Y written in language X

    Start with df, which contains image data (object Y) for template 1, and
    generate the corresponding audio and text data.
    """
    # assign a language X to each triple
    df["lang"] = get_language_column(args.n_per_language)

    # generate audio data
    df["audio_path"] = df.lang.apply(
        lambda x: sample_audio_file(x, args.commonvoice_dir, cv_df))

    # generate text data
    df["text"] = df.apply(lambda r: translate_text(r.class_name, r.lang, tr_client),
                          axis=1)

    df["template"] = 1
    return df


def template_2(args, tr_client, tts_client, df):
    """
    Template 2:
    - image: the flag of the country where language X is spoken
    - audio: a word Y spoken in any language other than X
    - text: the word Y written in language X

    Start with df, which contains text data (word Y) for template 2, and
    generate the corresponding image and audio data.
    """
    # assign a language X to each triple
    df["lang"] = get_language_column(args.n_per_language)

    # generate text data
    df["text"] = df.apply(lambda r: translate_text(r.word, r.lang, tr_client),
                          axis=1)

    # generate image data
    df["image_path"] = df.lang.apply(lambda x: args.flag_dir / ISO2FLAGFILE[x])

    # generate audio data
    df["audio_lang"] = df.lang.apply(sample_alternative_language)
    df["audio_path"] = df.apply(lambda r: generate_audio(r.word, r.audio_lang,
                                                         tr_client, tts_client,
                                                         args.audio_save_dir),
                                axis=1)

    df["template"] = 2
    return df


def template_3(args, tr_client, tts_client, df, cv_df):
    """
    Template 3:
    - image: an object Y
    - text: arbitrary text in language X
    - audio: the word for object Y spoken in language X

    Start with df, which contains image data (object Y) for template 3, and
    generate the corresponding audio and text data.
    """
    # assign a language X to each triple
    df["lang"] = get_language_column(args.n_per_language)

    # generate audio data
    df["audio_path"] = df.apply(lambda r: generate_audio(r.class_name, r.lang,
                                                         tr_client, tts_client,
                                                         args.audio_save_dir),
                                axis=1)

    # generate text data
    df["text"] = df.lang.apply(lambda x: sample_text_data(x, cv_df))

    df["template"] = 3
    return df


def template_4(args, tr_client, tts_client, df):
    """
    Template 4:
    - image: the flag of the country where language X is spoken
    - text: a word Y written in any language other than X
    - audio: the word Y spoken in language X

    Start with df, which contains text data (word Y) for template 4, and
    generate the corresponding image and audio data.
    """
    # assign a language X to each triple
    df["lang"] = get_language_column(args.n_per_language)

    # generate audio data
    df["audio_path"] = df.apply(lambda r: generate_audio(r.word, r.lang,
                                                         tr_client, tts_client,
                                                         args.audio_save_dir),
                                axis=1)

    # generate image data
    df["image_path"] = df.lang.apply(lambda x: args.flag_dir / ISO2FLAGFILE[x])

    # generate text data
    df["text_lang"] = df.lang.apply(sample_alternative_language)
    df["text"] = df.apply(lambda r: translate_text(r.word, r.text_lang, tr_client),
                          axis=1)

    df["template"] = 4
    return df


####################
# negative samples #
####################


def negative_samples(df):
    """
    Takes in the dataframe with positive samples and returns a dataframe with
    negative samples for support classification.

    For templates 1 and 2, we generate negative samples by fixing the text and
    shuffling the audio and image. For templates 3 and 4, we generate negative
    samples by fixing the audio and shuffling the text and image. In other words,
    the positive and negative samples have the same data, but how they are
    joined as triples is different.
    """
    df["audio_iso"] = df.apply(lambda r: audio_path_iso(r.audio_path, r.template),
                               axis=1)

    # TEMPLATE 1: fix text, shuffle audio and image
    t1 = df[df.template == 1]
    t1["audio_path"] = t1.apply(
        lambda r: sample_from_alternative_language(r.audio_iso,
                                                   t1[["audio_path", "audio_iso"]],
                                                   "audio_path"),
        axis=1)
    t1["image_path"] = t1.apply(lambda r: sample_negative(r.image_path, t1.image_path), axis=1)

    # TEMPLATE 2: fix text, shuffle audio and image
    t2 = df[df.template == 2]
    t2["audio_path"] = t2.apply(lambda r: sample_negative(r.audio_path, t2.audio_path), axis=1)
    t2["image_path"] = t2.image_path.apply(sample_negative_flag)

    # TEMPLATE 3: fix audio, shuffle text and image
    t3 = df[df.template == 3]
    t3["text"] = t3.apply(
        lambda r: sample_from_alternative_language(r.audio_iso,
                                                   t3[["text", "audio_iso"]],
                                                   "text"),
        axis=1)
    t3["image_path"] = t3.apply(lambda r: sample_negative(r.image_path, t3.image_path), axis=1)

    # TEMPLATE 4: fix audio, shuffle text and image
    t4 = df[df.template == 4]
    t4["text"] = t4.apply(lambda r: sample_negative(r.text, t4.text), axis=1)
    t4["image_path"] = t4.image_path.apply(sample_negative_flag)

    df_neg = pd.concat([t1, t2, t3, t4], ignore_index=True)
    return df_neg[["text", "audio_path", "image_path", "template"]]


if __name__ == '__main__':
    args = parse_args_generate_data()

    tr_client = translate.Client()
    tts_client = texttospeech.TextToSpeechClient()

    # get common voice data for template 1 (for audio) and template 3 (for text)
    cv_df = get_commonvoice_data(args.commonvoice_dir, args.commonvoice_split)

    # sample image data for templates 1 and 3 from appropriate dataset split
    img_df = pd.read_csv(args.image_path) \
               .sample(n=args.n_per_language * len(LANGUAGES) * 2,
                       ignore_index=True)
    img_df_t1, img_df_t3 = train_test_split(img_df, train_size=0.5, shuffle=True)

    # sample text data for templates 2 and 4 from appropriate dataset split
    n_words = args.n_per_language * len(LANGUAGES) * 2
    txt_df = pd.read_csv(args.text_path) \
               .sort_values(by=['count'], ascending=False) \
               .head(n_words)
    assert len(txt_df) == n_words, "Too few words in this split for n_per_language."
    txt_df_t2, txt_df_t4 = train_test_split(txt_df, train_size=0.5, shuffle=True)

    # generate data for each template
    t1 = template_1(args, tr_client, img_df_t1, cv_df) \
            [["text", "audio_path", "image_path", "template"]]
    t2 = template_2(args, tr_client, tts_client, txt_df_t2) \
            [["text", "audio_path", "image_path", "template"]]
    t3 = template_3(args, tr_client, tts_client, img_df_t3, cv_df) \
            [["text", "audio_path", "image_path", "template"]]
    t4 = template_4(args, tr_client, tts_client, txt_df_t4) \
            [["text", "audio_path", "image_path", "template"]]
    df = pd.concat([t1, t2, t3, t4], ignore_index=True)

    # get negative samples if generating data for support classification
    if args.negative_samples:
        df_neg = negative_samples(df) \
                    [["text", "audio_path", "image_path", "template"]]

        df["in_support"] = 1
        df_neg["in_support"] = 0

        df = pd.concat([df, df_neg], ignore_index=True)

    df.to_csv(args.save_path, index=False)