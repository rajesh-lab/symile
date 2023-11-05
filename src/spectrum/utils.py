import pandas as pd
import torch


def get_test_distribution(dm):
    """
    Returns a dataframe that, for each vector (v_a, v_b, v_c) in the test,
    calculates the percent of the vector that belongs to each possible value.
    The possible values for vectors with dimensionality 1 are [0, 1].
    The possible values for vectors with dimensionality 2 are [00, 01, 10, 11].

    Args:
        dm (SyntheticDataModule): data module.
    Returns:
        df (pd.DataFrame): with the following columns:
            "value": e.g. "00", "01" for two dims, or "0", "1" for one dim
            "count": number of occurrences of "value" in that vector (a, b, c)
            "type": which vector (a, b, c) the "value" belongs to
            "prob": "count" / total number of elements in the vector
    """
    v_a = dm.ds_test.v_a
    v_b = dm.ds_test.v_b
    v_c = dm.ds_test.v_c
    data = {"a": v_a, "b": v_b, "c": v_c}

    df_dict = {"value": [], "count": [], "type": []}
    for m in ["a", "b", "c"]:
        counts = torch.unique(data[m], dim=0, return_counts=True)
        values = []
        for i in range(len(counts[0])):
            if v_a.shape[1] == 1: # d_v = 1
                value = str(int(counts[0][i]))
            elif v_a.shape[1] == 2: # d_v = 2
                value = ""
                for j in range(2):
                    value += str(int(counts[0][i][j]))
            values.append(value)
        df_dict["value"].extend(values)
        df_dict["count"].extend(counts[1].tolist())
        df_dict["type"].extend(m*len(counts[0]))

    df = pd.DataFrame(df_dict)
    df["prob"] = df["count"] / len(v_a)

    return df