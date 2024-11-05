"""
Script to split a dataframe (e.g. train.csv) into sub-dataframes (e.g.
train0.csv, train1.csv, etc.) so that save_representations.py can be run in
parallel on each sub-dataframe.
"""
import pandas as pd

from args import parse_args_split_df

if __name__ == '__main__':
    args = parse_args_split_df()

    df = pd.read_csv(args.csv_to_split)

    # split df into n sub-dfs so that each has size k
    k = args.sub_df_size
    n = len(df) // k
    num_sub_dfs = n
    len_sub_dfs = 0

    for i in range(n):
        sub_df = df.iloc[i*k:(i+1)*k]
        sub_df_name = f"{args.csv_to_split.stem}{i}"

        sub_df.to_csv(args.save_dir / f"{sub_df_name}.csv", index=False)

        len_sub_dfs += len(sub_df)

    if len(df) % k != 0:
        sub_df = df.iloc[n*k:]
        sub_df_name = f"{args.csv_to_split.stem}{n}"

        sub_df.to_csv(args.save_dir / f"{sub_df_name}.csv", index=False)

        len_sub_dfs += len(sub_df)
        num_sub_dfs = n + 1

    print(f"Split df into {num_sub_dfs} sub-dfs. Original df had {len(df)} rows. Sub-dfs have {len_sub_dfs} rows in total. (These two numbers should match.)")