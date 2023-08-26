import os
import pandas as pd
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:
    wandb = None

from args import parse_args_main
from datasets import SymileDataset
from models import AudioEncoder, ImageEncoder, TextEncoder, SymileModel
from src.losses import pairwise_infonce, symile
from utils import seed_all, wandb_init


def load_data(args):
    df = pd.read_csv(args.dataset_path)
    df["text"] = df.text.fillna("")
    ds = SymileDataset(args, df)
    return DataLoader(ds, batch_size=args.batch_sz, shuffle=True)

def pretrain(args, symile_model):
    dl = load_data(args)
    loss_fn = symile if args.loss_fn == "symile" else pairwise_infonce
    # TODO: make sure these are all of the parameters that need to be optimized
    optimizer = torch.optim.AdamW(symile_model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for data in dl:
            breakpoint()
            # TODO: questions asked to AP
            r_a, r_i, r_t, logit_scale_exp = symile_model(**data)
            audio = audio_encoder(audio)
            image = image_encoder(**data["image"])
            text = text_encoder(**data["text"])

            loss = loss_fn(r_a, r_b, r_c, logit_scale_exp, args.normalize)

if __name__ == '__main__':
    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'
    args = parse_args_main()
    wandb_init(args)
    if args.use_seed:
        seed_all(args.seed)

    # PRETRAIN
    print("\n\n\n...pretraining...\n")
    audio_encoder = AudioEncoder(args.audio_model_id)
    image_encoder = ImageEncoder(args.image_model_id)
    text_encoder = TextEncoder(args.text_model_id)
    symile_model = SymileModel(audio_encoder, image_encoder, text_encoder,
                               args.logit_scale_init)
    pretrain(args, symile_model)
    # TODO: make sure that at the end of this, all the encoders have actually
    # been trained (weights changed)

    # EVALUATE