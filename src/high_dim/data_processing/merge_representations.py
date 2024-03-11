import os

import torch

from args import parse_args_merge_representations

if __name__ == '__main__':
    args = parse_args_merge_representations()

    audio = torch.empty(0)
    image = torch.empty(0)
    text_attention_mask = torch.empty(0)
    text_input_ids = torch.empty(0)
    cls_id = torch.empty(0)
    idx = torch.empty(0)
    cls = []
    lang = []
    words = []

    for i in range(args.num_subdirs):
        subdir = args.data_dir / f"train{i}"
        audio = torch.cat((audio, torch.load(subdir / f"audio_train{i}.pt")))
        image = torch.cat((image, torch.load(subdir / f"image_train{i}.pt")))
        text_attention_mask = torch.cat((text_attention_mask, torch.load(subdir / f"text_attention_mask_train{i}.pt")))
        text_input_ids = torch.cat((text_input_ids, torch.load(subdir / f"text_input_ids_train{i}.pt")))
        cls_id = torch.cat((cls_id, torch.load(subdir / f"cls_id_train{i}.pt")))
        idx = torch.cat((idx, torch.load(subdir / f"idx_train{i}.pt")))

        with open(subdir / f"cls_train{i}.txt", "r") as f:
            cls.extend(f.read().splitlines())
        with open(subdir / f"lang_train{i}.txt", "r") as f:
            lang.extend(f.read().splitlines())
        with open(subdir / f"words_train{i}.txt", "r") as f:
            words.extend(f.read().splitlines())

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    torch.save(audio, args.save_dir / "audio_train.pt")
    torch.save(image, args.save_dir / "image_train.pt")
    torch.save(text_attention_mask, args.save_dir / "text_attention_mask_train.pt")
    torch.save(text_input_ids, args.save_dir / "text_input_ids_train.pt")
    torch.save(cls_id, args.save_dir / "cls_id_train.pt")
    torch.save(idx, args.save_dir / "idx_train.pt")

    with open(args.save_dir / "cls_train.txt", "w") as f:
        f.write("\n".join(cls))
    with open(args.save_dir / "lang_train.txt", "w") as f:
        f.write("\n".join(lang))
    with open(args.save_dir / "words_train.txt", "w") as f:
        f.write("\n".join(words))