"""
This script takes in a json file as an argument, and adds any missing translations
of the ImageNet class names to the file.

We use images from the ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
2012-2017. The train set has 1,281,167 images from 1,000 categories.
"""
import json
import os

import pandas as pd

from google.cloud import translate_v2 as translate

from args import parse_args_source_data
from constants import ALL_LANGUAGES


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


def get_new_data_ref(data_ref, classmapping_path):
    """
    `data_ref` is a dictionary whose keys are ImageNet class names and whose
    values are also dictionaries. For example, data_ref["butterfly"] is
        {
            "synset_id": "n02279972",
            "ar": "فراشة",
            "en": "butterfly",
            "el": "πεταλούδα",
            "hi": "तितली",
            "ja": "蝶"
        }

    This function goes through each class name in `classmapping_path` and adds
    any necessary translations for the languages in ALL_LANGUAGES. It then
    returns a new data_ref.

    Args:
        data_ref (dict): Dictionary whose keys are ImageNet class names. The
                         value associated with each name is a language ISO-639 codes with
                         values "synset_id" and
        classmapping_path (Path): Path to ImageNet synset mapping txt file.
    """
    tr_client = translate.Client()

    class_mappings = get_class_mappings(classmapping_path)

    for ix, r in class_mappings.iterrows():
        if ix % 10 == 0:
            print(f"  working on class_mappings row {ix}...")

        if r.class_name in data_ref:

            assert "synset_id" in data_ref[r.class_name], \
                f"synset_id is not in data_ref[{r.class_name}]"

            for l in ALL_LANGUAGES:
                if l not in data_ref[r.class_name]:
                    data_ref[r.class_name][l] = translate_text(r.class_name, l, tr_client)
        else:
            data_ref[r.class_name] = {"synset_id": r.class_id}
            for l in ALL_LANGUAGES:
                data_ref[r.class_name][l] = translate_text(r.class_name, l, tr_client)
    return data_ref


if __name__ == '__main__':
    args = parse_args_source_data()

    if os.path.exists(args.data_reference):
        data_ref = json.load(open(args.data_reference))
    else:
        data_ref = {}

    new_data_ref = get_new_data_ref(data_ref, args.imagenet_classmapping_path)

    with open(args.data_reference, 'w') as f:
        # overwrite data_ref
        json.dump(new_data_ref, f, ensure_ascii=False, indent=4)