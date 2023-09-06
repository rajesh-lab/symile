"""
TODO:
- clean up this whole file
- Include instructions for how to get google translate/tts up and running
- Probably move some of these functions in the utils file
- Depending on how you structure this, you may want to move all the tr and tts clients
- you may not even need the template information after all...
"""

import os
from pathlib import Path
import random

import numpy as np
import pandas as pd

from google.cloud import texttospeech
from google.cloud import translate_v2 as translate

from args import parse_args_generate_data
from constants import *

def get_language_column(n_per_language):
    lang_col = [[x] * n_per_language for x in LANGUAGES]
    lang_col = [x for sublist in lang_col for x in sublist]
    return random.sample(lang_col, len(lang_col))

def sample_data(df, n_per_language):
    return df.sample(n=n_per_language*len(LANGUAGES), ignore_index=True)

def translate_text(text, language, tr_client):
    if language == "english":
        return text
    else:
        return tr_client.translate(text, target_language=LANG2ISOCODE[language])['translatedText']

def synsetmapping_to_name(synset_str):
    synset_str = synset_str.split(" ")[1:]
    return " ".join(synset_str).split(",")[0]

def get_class_mappings(args):
    class_mapping = pd.read_csv(args.imagenet_dir / args.imagenet_classmapping_filename,
                                sep="\t", names=["synset"])
    class_mapping["class_id"] = class_mapping.synset.apply(lambda x: x.split(" ")[0])
    class_mapping["class_name"] = class_mapping.synset.apply(synsetmapping_to_name)
    return class_mapping[["class_id", "class_name"]]

def predstr_to_class(class_string):
    class_string = class_string.split(" ")
    classes = []
    for x in class_string:
        if len(x) > 0 and x[0] == "n":
            classes.append(x)
    classes = list(set(classes))
    if len(classes) == 1:
        return classes[0]
    else:
        return np.nan

def get_image_path(dir, cls, img_id):
    filename = Path(img_id + ".JPEG")
    return dir / Path("ILSVRC/Data/CLS-LOC/train") / cls / filename

def get_image_data(args):
    class_mapping = get_class_mappings(args)
    df = pd.read_csv(args.imagenet_dir / args.imagenet_train_filename)
    df["class_id"] = df.PredictionString.apply(predstr_to_class)
    df = df.rename(columns={"ImageId": "img_id"}).\
            drop(columns=["PredictionString"]).\
            join(class_mapping.set_index("class_id"), on="class_id").\
            dropna()
    df["image_path"] = df.apply(
        lambda row: get_image_path(args.imagenet_dir, row.class_id, row.img_id),
        axis=1)
    return df

def generate_audio(text_english, language, tr_client, tts_client, audio_save_dir):
    """
    right now the std voices are being used
    """
    Path(audio_save_dir).mkdir(parents=True, exist_ok=True)
    save_path = audio_save_dir / f"{language}_{text_english}.mp3"

    if not os.path.exists(save_path):
        text = translate_text(text_english, language, tr_client)

        voice = texttospeech.VoiceSelectionParams(
            language_code=LANG2LANGCODE[language], name=LANG2VOICES[language][0]
        )

        response = tts_client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=voice,
            audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        )

        with open(save_path, "wb") as out:
            out.write(response.audio_content)

    return save_path

def get_word_data(n_per_language):
    """
    https://norvig.com/ngrams/count_1w.txt
    hardcoded right now but will need to change
    """
    df = pd.read_csv('words.txt', sep='\t', names=['text_english', 'count'])

    # filter data
    df = df[df.text_english.str.len() > 2].drop_duplicates(subset=['text_english'])

    # get most common words
    n_words = n_per_language * 5
    assert len(df) >= n_words, "There are not enough words in the dataframe for all 5 languages."
    df = df.sort_values(by=['count'], ascending=False).head(n_words)
    return df[['text_english']]

def sample_alternative_language(x):
    """
    Sample a language Z from the other four languages.
    """
    return np.random.choice([l for l in LANGUAGES if l != x])

##############
# template 1 #
##############

def sample_audio_file(lang, commonvoice_dir):
    commonvoice_dir = Path(commonvoice_dir / LANG2ISOCODE[lang] / "clips")
    return commonvoice_dir / random.choice(os.listdir(commonvoice_dir))

def template_1(args):
    tr_client = translate.Client()
    tts_client = texttospeech.TextToSpeechClient()

    # MODALITY: IMAGE
    # TODO: template 3 uses this...maybe pull this function out so you're not pulling the image data twice?
    df = get_image_data(args)
    df = sample_data(df, args.n_per_language)

    # MODALITY: TEXT
    df["lang"] = get_language_column(args.n_per_language)
    df["text"] = df.apply(lambda row: translate_text(row.class_name, row.lang, tr_client),
                          axis=1)

    # MODALITY: AUDIO
    df["audio_path"] = df.lang.apply(lambda x: sample_audio_file(x, args.commonvoice_dir))

    df["template"] = 1
    return df

##############
# template 2 #
##############

def template_2(args):
    tr_client = translate.Client()
    tts_client = texttospeech.TextToSpeechClient()

    # MODALITY: TEXT
    df = get_word_data(args.n_per_language)
    # technically unnecessary now, but eventually will need to update get_word_data()
    df = sample_data(df, args.n_per_language)
    df["text_lang"] = get_language_column(args.n_per_language)
    df["text"] = df.apply(lambda row: translate_text(row.text_english, row.text_lang, tr_client),
                          axis=1)

    # MODALITY: IMAGE
    df["image_path"] = df.text_lang.apply(lambda x: args.flag_dir / LANG2FLAGFILE[x])

    # MODALITY: AUDIO
    df["audio_lang"] = df.text_lang.apply(sample_alternative_language)
    df["audio_path"] = df.apply(lambda row: generate_audio(row.text_english,
                                                           row.audio_lang,
                                                           tr_client,
                                                           tts_client,
                                                           args.audio_save_dir),
                                axis=1)

    df["template"] = 2
    return df

##############
# template 3 #
##############

def get_text_data(commonvoice_dir):
    dfs = []
    for lang in LANGUAGES:
        df = pd.read_csv(commonvoice_dir / LANG2ISOCODE[lang] / "train.tsv", sep="\t")[["sentence"]]
        df["lang"] = lang
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def sample_text_data(lang, text_df):
    return text_df[text_df.lang == lang].sample(n=1).iloc[0].sentence

def template_3(args):
    tr_client = translate.Client()
    tts_client = texttospeech.TextToSpeechClient()

    # MODALITY: IMAGE
    # TODO: template 1 uses this...maybe pull this function out so you're not pulling the image data twice?
    df = get_image_data(args)
    df = sample_data(df, args.n_per_language)

    # MODALITY: AUDIO
    df["lang"] = get_language_column(args.n_per_language)
    df["audio_path"] = df.apply(lambda row: generate_audio(row.class_name,
                                                           row.lang,
                                                           tr_client,
                                                           tts_client,
                                                           args.audio_save_dir),
                                axis=1)

    # MODALITY: TEXT
    text_data = get_text_data(args.commonvoice_dir)
    df["text"] = df.lang.apply(lambda x: sample_text_data(x, text_data))

    df["template"] = 3
    return df

##############
# template 4 #
##############

def template_4(args):
    tr_client = translate.Client()
    tts_client = texttospeech.TextToSpeechClient()

    # MODALITY: AUDIO
    df = get_word_data(args.n_per_language)
    # technically unnecessary now, but eventually will need to update get_word_data()
    df = sample_data(df, args.n_per_language)
    df["audio_lang"] = get_language_column(args.n_per_language)
    df["audio_path"] = df.apply(lambda row: generate_audio(row.text_english,
                                                           row.audio_lang,
                                                           tr_client,
                                                           tts_client,
                                                           args.audio_save_dir),
                                axis=1)

    # MODALITY: IMAGE
    df["image_path"] = df.audio_lang.apply(lambda x: args.flag_dir / LANG2FLAGFILE[x])

    # MODALITY: AUDIO
    df["text_lang"] = df.audio_lang.apply(sample_alternative_language)
    df["text"] = df.apply(lambda row: translate_text(row.text_english, row.text_lang, tr_client),
                          axis=1)

    df["template"] = 4
    return df

####################
# negative samples #
####################

def audio_path_iso(audio_path, template):
    if template == 1:
        iso = Path(audio_path).stem.split("_")[-2]
    else:
        iso = LANG2ISOCODE[Path(audio_path).stem.split("_")[0]]
    assert iso in LANG2ISOCODE.values(), f"{iso} is not a valid ISO code."
    return iso

def sample_negative_from_diff_language(row_iso, all_rows, col):
    """used for audio t1 and text t3 where language needs to be diff."""
    neg = all_rows[all_rows.audio_iso != row_iso].sample()
    assert row_iso != neg.audio_iso.item(), \
        "Negative sample must be in a different language."
    return neg[col].item()

def sample_negative(row_path, all_paths):
    neg_path = all_paths[all_paths != row_path].sample().item()
    assert row_path != neg_path, \
        f"Negative sample {neg_path} must be different than {row_path}."
    return neg_path

def sample_negative_flag(row_path):
    """this is used for t2 and t4"""
    row_flag = Path(row_path).name
    flags = list(LANG2FLAGFILE.values())
    flags.remove(row_flag)
    neg_flag = random.choice(flags)
    assert row_flag != neg_flag, \
        f"Negative flag {neg_flag} must be different than {row_flag}."
    return Path(row_path).parent / neg_flag

def negative_samples(df):
    df["audio_iso"] = df.apply(lambda r: audio_path_iso(r.audio_path, r.template), axis=1)

    # TEMPLATE 1: fix text, shuffle audio and image
    t1 = df[df.template == 1]
    t1["audio_path"] = t1.apply(
        lambda r: sample_negative_from_diff_language(r.audio_iso,
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
        lambda r: sample_negative_from_diff_language(r.audio_iso,
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

    t1 = template_1(args)[["text", "audio_path", "image_path", "template"]]
    t2 = template_2(args)[["text", "audio_path", "image_path", "template"]]
    t3 = template_3(args)[["text", "audio_path", "image_path", "template"]]
    t4 = template_4(args)[["text", "audio_path", "image_path", "template"]]
    df = pd.concat([t1, t2, t3, t4], ignore_index=True)

    if args.negative_samples:
        df_neg = negative_samples(df)

        df["in_support"] = 1
        df_neg["in_support"] = 0

        df = pd.concat([df, df_neg], ignore_index=True)

    df.to_csv(args.save_path, index=False)