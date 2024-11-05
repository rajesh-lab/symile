"""
This script takes in a json file as an argument, and adds any missing translations
of the ImageNet class names to the file.
"""
import html
import json
import os

import pandas as pd

from google.cloud import translate_v2 as translate

from args import parse_args_generate_translations
from constants import LANGUAGES_10


def manual_translation(args, translations_dict):
    """
    Updates Google Translate API translations with manual translations (both
    to make corrections and to avoid overlap between languages.)
    """
    manual_translations = json.load(open(args.manual_translations_path))

    for class_name, translations in manual_translations.items():
        synset_id = translations.get('synset_id')

        if class_name in translations_dict and translations_dict[class_name].get('synset_id') == synset_id:
            for lang, translation in translations.items():
                if lang != 'synset_id':
                    # replace translation in translations_dict with translation from manual_translations
                    translations_dict[class_name][lang] = translation
        else:
            ValueError(f"Either {key} was not found in translations_dict or there's a synset_id mismatch.")

    return translations_dict


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

    df = pd.read_csv(mapping_path, sep="\t", names=["synset"])
    df["class_id"] = df.synset.apply(lambda x: x.split(" ")[0])
    df["class_name"] = df.synset.apply(_synsetmapping_to_name)

    # manually distinguish between homonyms

    # if class_id = n03126707, then class_name = "crane machine"
    df.loc[df["class_id"] == "n03126707", "class_name"] = "crane machine"
    # if class_id = n02012849, then class_name = "crane bird"
    df.loc[df["class_id"] == "n02012849", "class_name"] = "crane bird"

    # both class_id n03710637 and n03710721 are `maillot`, so we manually set
    # class_id n03710721 to `swimsuit`
    df.loc[df["class_id"] == "n03710721", "class_name"] = "swimsuit"

    return df[["class_id", "class_name"]]


def translate_text(text, lang, tr_client):
    """
    Uses Google Translate API to translate `text` to specified `lang` (iso code).
    Since class names are already in English, do not translate if `lang`
    is English.
    """
    if lang == "en":
        return text
    else:
        try:
            tr_client.translate(text, target_language=lang)['translatedText']
        except:
            print(f"  Error translating {text} to {lang}")
            return text
        return tr_client.translate(text, target_language=lang)['translatedText']


def get_translations(translations_dict, classmapping_path):
    """
    `translations_dict` is a dictionary whose keys are ImageNet class names and whose
    values are also dictionaries. For example, translations_dict["butterfly"] is
        {
            "synset_id": "n02279972",
            "ar": "فراشة",
            "en": "butterfly",
            "el": "πεταλούδα",
            "hi": "तितली",
            "ja": "蝶"
        }

    This function goes through each class name in `classmapping_path` and adds
    any necessary translations for the languages in LANGUAGES_10. It then
    returns a new translations_dict.

    Args:
        translations_dict (dict): Dictionary whose keys are ImageNet class names. The
                         value associated with each name is a language ISO-639 codes with
                         values "synset_id" and
        classmapping_path (Path): Path to ImageNet synset mapping txt file.
    """
    tr_client = translate.Client()

    class_mappings = get_class_mappings(classmapping_path)

    for ix, r in class_mappings.iterrows():
        if ix % 10 == 0:
            print(f"  working on class_mappings row {ix}...")

        if r.class_name in translations_dict:

            assert "synset_id" in translations_dict[r.class_name], \
                f"synset_id is not in translations_dict[{r.class_name}]"

            for l in LANGUAGES_10:
                if l not in translations_dict[r.class_name]:
                    # html.unescape converts character references e.g. &#39; to Unicode
                    translation = html.unescape(translate_text(r.class_name, l, tr_client))

                    translations_dict[r.class_name][l] = translation
        else:
            translations_dict[r.class_name] = {"synset_id": r.class_id}
            for l in LANGUAGES_10:
                # html.unescape converts character references e.g. &#39; to Unicode
                translation = html.unescape(translate_text(r.class_name, l, tr_client))

                translations_dict[r.class_name][l] = translation
    return translations_dict


def get_cls_ids(translations_dict):
    for ix, (class_name, details) in enumerate(translations_dict.items()):
        details["cls_id"] = ix
    return translations_dict


if __name__ == '__main__':
    args = parse_args_generate_translations()

    if os.path.exists(args.translations_path):
        translations_dict = json.load(open(args.translations_path))
    else:
        translations_dict = {}

    translations_dict = get_translations(translations_dict, args.imagenet_classmapping_path)

    translations_dict = manual_translation(args, translations_dict)

    translations_dict = get_cls_ids(translations_dict)

    with open(args.translations_path, 'w') as f:
        # overwrite translations_dict
        json.dump(translations_dict, f, ensure_ascii=False, indent=4)