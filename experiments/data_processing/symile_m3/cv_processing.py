"""
This script removes any audio clips that have duration 0.0 seconds.
"""
import os

import pandas as pd

from args import parse_args_generate_data
from constants import LANGUAGES_10

if __name__ == '__main__':
    args = parse_args_generate_data()

    for lang in LANGUAGES_10:
        print(f"Working on {lang}...")

        pt = args.cv_dir / f"cv/{lang}/clip_durations.tsv"
        df = pd.read_csv(pt, sep="\t")
        df.columns = ["file", "duration"]

        for ix, r in df.iterrows():
            if r["duration"] == 0:
                file_pt = args.cv_dir / f"cv/{lang}/clips" / r["file"]
                if os.path.exists(file_pt):
                    print(f"Removing {file_pt}")
                    os.remove(file_pt)