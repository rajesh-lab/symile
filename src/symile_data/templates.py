"""TODO: clean up this whole file"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import random

from google.cloud import texttospeech
from google.cloud import translate_v2 as translate

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
    breakpoint()
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

# for each language X
    # get images (5K - make a variable)
    # get class names for those images
    # get the text for each class name in language X
    # get random audio clips of languauge X being spoken
# return: probably a dataframe of paths, except for the text?


def sample_audio_file(lang, commonvoice_dir):
    commonvoice_dir = Path(commonvoice_dir / lang / "clips")
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
    return df


##############
# template 3 #
##############

def get_text_data(commonvoice_dir):
    dfs = []
    for lang in LANGUAGES:
        df = pd.read_csv(commonvoice_dir / lang / "train.tsv", sep="\t")[["sentence"]]
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
    return df


if __name__ == "__main__":
    # TODO: FIX ALL ARGS
    # TODO: INCLUDE INSTUCTIONS for how to get google translate/tts up and running
    # TODO: Probably move some of these functions in the utils file
    # Depending on how you structure this, you may want to move all the tr and tts clients
    # here instead of in the functions? so you only establish them once?

    args = {
        "n_per_language": 1,
        "audio_save_dir": Path("./audio"),
        "flag_dir": Path("/Users/adrielsaporta/Documents/NYU/symile_data/flags"),
        "imagenet_dir": Path("/Users/adrielsaporta/Documents/NYU/symile_data/imagenet/imagenet-object-localization-challenge"),
        "imagenet_classmapping_filename": Path("LOC_synset_mapping.txt"),
        "imagenet_train_filename": Path("LOC_train_solution.csv"),
        "commonvoice_dir": Path("/Users/adrielsaporta/Documents/NYU/symile_data/common_voice")
    }

    df = template_1(args)
    breakpoint()
    # template_2(args)
    # template_3(args)
    # template_4(args)