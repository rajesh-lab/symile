"""
We use images from the ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
2012-2017. The train set has 1,281,167 images from 1,000 categories. The val set
has 50,000 images with associated categories. The test set has 100,000 images,
but without associated categories. Therefore, this script divides the ImageNet
splits as follows:

- ImageNet train set divided into a train/val splits for pretraining.
- ImageNet val set divided into train/val/test splits for support
  classification.
- ImageNet val set used for zero-shot classification.

Final datasets are created in `generate_data.py` by sampling the desired number
of data points from the respective splits generated above.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from args import parse_args_create_image_splits
from utils import get_splits, split_size


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


def predstr_to_class(class_str):
    """
    Because `args.imagenet_val_path` contains bounding box annotations in
    addition to the classifcation annotation, `class_str` is in the form
    `n03109150 0 14 216 299`. Sometimes there are two bounding boxes per image,
    and so `class_str` is in the form
    `n06874185 103 115 154 218 n06874185 346 116 397 220`.

    This function takes `class_str` for a given data point and finds the
    class_ids associated with its bounding boxes. If there is only one class_id,
    that class_id is returned. If there are multiple class_ids for this image,
    then np.nan is returned.
    """
    class_str = class_str.split(" ")
    classes = []
    for x in class_str:
        if len(x) > 0 and x[0] == "n":
            classes.append(x)
    classes = list(set(classes))
    if len(classes) == 1:
        return classes[0]
    else:
        return np.nan


def add_image_paths(df, imagenet_dir, split):
    """
    ImageNet saves training images under the folders with the names of their
    synsets, and saves validation images all in the same folder.

    Args:
        df (pd.DataFrame): columns are `class_id` and `img_id`.
        imagenet_dir (Path): Where ImageNet image data is saved (e.g.
                             `/imagenet/ILSVRC/Data/CLS-LOC`).
        split (str): ImageNet split ("train" or "val").
    Returns:
        (pd.DataFrame): df with added column `image_path`.
    """
    def _image_path(class_id, img_id):
        img_dir = imagenet_dir / Path(split)
        filename = Path(img_id + ".JPEG")
        if split == "train":
            return img_dir / class_id / filename
        elif split == "val":
            return img_dir / filename

    df["image_path"] = df.apply(lambda r: _image_path(r.class_id, r.img_id),
                                axis=1)
    return df


if __name__ == '__main__':
    args = parse_args_create_image_splits()

    class_mapping = get_class_mappings(args.imagenet_classmapping_path)

    # get pre-training train/val split from ImageNet train set
    df = pd.read_csv(args.imagenet_train_path, delim_whitespace=True, header=None)
    df = df.iloc[:, 0].str.split("/", expand=True)
    df = df.rename(columns={df.columns[0]: "class_id", df.columns[1]: "img_id"}) \
           .join(class_mapping.set_index("class_id"), on="class_id") \
           .dropna()
    imagenet_train = add_image_paths(df, args.imagenet_dir, "train")

    pretrain_train, pretrain_val = train_test_split(imagenet_train,
                                                    train_size=split_size(args.pretrain_train_size),
                                                    shuffle=True)

    pretrain_train.to_csv(args.save_path / "img_pretrain_train.csv", index=False)
    pretrain_val.to_csv(args.save_path / "img_pretrain_val.csv", index=False)
    print(f"Pretrain train size: {len(pretrain_train)}")
    print(f"Pretrain val size: {len(pretrain_val)}")

    # get support classification train/val/test splits from ImageNet val set ###
    df = pd.read_csv(args.imagenet_val_path)
    df.PredictionString = df.PredictionString.apply(predstr_to_class)
    df = df.rename(columns={"PredictionString": "class_id", "ImageId": "img_id"}) \
           .join(class_mapping.set_index("class_id"), on="class_id") \
           .dropna()
    imagenet_val = add_image_paths(df, args.imagenet_dir, "val")

    support_train, support_val, support_test = \
        get_splits(imagenet_val, args.support_train_size, args.support_val_size)

    support_train.to_csv(args.save_path / "img_support_train.csv", index=False)
    support_val.to_csv(args.save_path / "img_support_val.csv", index=False)
    support_test.to_csv(args.save_path / "img_support_test.csv", index=False)
    print(f"Support train size: {len(support_train)}")
    print(f"Support val size: {len(support_val)}")
    print(f"Support test size: {len(support_test)}")

    # get zeroshot classification test split from ImageNet val set ###
    imagenet_val.to_csv(args.save_path / "img_zeroshot_test.csv", index=False)
    print(f"Zeroshot test size: {len(imagenet_val)}")