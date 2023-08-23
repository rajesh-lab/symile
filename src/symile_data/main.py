try:
    import wandb
except ImportError:
    wandb = None

from args import parse_args
from datasets import SymileDataset

if __name__ == '__main__':
    args = parse_args()

    dataset = SymileDataset(args)