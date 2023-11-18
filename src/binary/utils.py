import pandas as pd
import plotly.express as px
import torch

from information_measures import prob_c_1d, prob_c_given_a_b_1d, \
                                 prob_c_2d, prob_c_given_a_b_2d


def save_test_distribution(dm, save_dir, loss_fn, i_p):
    """
    Returns a dataframe that, for each vector (v_a, v_b, v_c) in the test,
    calculates the percent of the vector that belongs to each possible value.
    The possible values for vectors with dimensionality 1 are [0, 1].
    The possible values for vectors with dimensionality 2 are [00, 01, 10, 11].

    Args:
        dm (SyntheticDataModule): data module.
        save_dir (Path): i_p-specific directory to save dataframe and plot to.
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

    df.to_csv(save_dir / f"test_dist_{loss_fn}_i_p_{i_p}.csv", index=False)
    fig = px.line(df, x="value", y="prob", color="type")
    fig.write_image(save_dir / f"test_dist_{loss_fn}_i_p_{i_p}.png")


def likelihood_ratios(dim):
    """
    Calculate the true likelihood ratio p(a,b,c)/p(a)p(b)p(c) = p(c|a,b)/p(c)
    for all possible values of a, b, c and all possible values of i_p.
    """
    data = {}
    for i_p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        data[i_p] = {"abc": [], "value": []}

        if dim == 1:
            for a in [0, 1]:
                for b in [0, 1]:
                    for c in [0, 1]:
                        p_c_given_a_b = prob_c_given_a_b_1d(a, b, c, i_p)
                        p_c = prob_c_1d(c, i_p)
                        lr = p_c_given_a_b / p_c if p_c != 0 else 0
                        data[i_p]["abc"].append(str(a)+str(b)+str(c))
                        data[i_p]["value"].append(lr)

        elif dim == 2:
            for a_1 in [0, 1]:
                for a_2 in [0, 1]:
                    for b_1 in [0, 1]:
                        for b_2 in [0, 1]:
                            for c_1 in [0, 1]:
                                for c_2 in [0, 1]:
                                    p_c_given_a_b = prob_c_given_a_b_2d(
                                        a_1, a_2, b_1, b_2, c_1, c_2, i_p
                                    )
                                    p_c = prob_c_2d(c_1, c_2, i_p)
                                    lr = p_c_given_a_b / p_c if p_c != 0 else 0
                                    data[i_p]["abc"].append(
                                        str(a_1)+str(a_2)+str(b_1)+str(b_2)+str(c_1)+str(c_2)
                                    )
                                    data[i_p]["value"].append(lr)
    return data


def all_data_vectors(dim):
    v_a = []
    v_b = []
    v_c = []
    abc = []
    if dim == 1:
        values = [0, 1]
        for a in values:
            for b in values:
                for c in values:
                    abc.append(str(a) + str(b) + str(c))
                    v_a.append([a])
                    v_b.append([b])
                    v_c.append([c])
    elif dim == 2:
        values = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for a in values:
            for b in values:
                for c in values:
                    abc.append("".join([str(i) for i in a]+[str(i) for i in b]+[str(i) for i in c]))
                    v_a.append(a)
                    v_b.append(b)
                    v_c.append(c)
    v_a = torch.Tensor(v_a).to(torch.float32)
    v_b = torch.Tensor(v_b).to(torch.float32)
    v_c = torch.Tensor(v_c).to(torch.float32)
    return v_a, v_b, v_c, abc


def save_likelihood_ratio_vs_score(i_p, loss_fn, model, lr_data, save_dir, dim):
    """
    For each i_p, create and save a plot that compares the likelihood ratio to
    the multilinear inner product for different values of a, b, c.
    """
    lr_data["type"] = ["likelihood ratio"] * len(lr_data["value"])

    # get representations
    v_a, v_b, v_c, abc = all_data_vectors(dim=dim)
    r_a, r_b, r_c = model.encoders(v_a, v_b, v_c)

    # calculate score
    if loss_fn == "symile":
        score_fn = torch.sum(r_a * r_b * r_c, axis=1)
    elif loss_fn == "pairwise_infonce":
        score_fn = torch.sum((r_a * r_b) + (r_b * r_c) + (r_a * r_c), axis=1)

    # save plot
    lr_data["abc"] += abc
    lr_data["value"] += score_fn.tolist()
    lr_data["type"] += ["score fn"] * len(abc)
    df = pd.DataFrame(lr_data)
    df.to_csv(save_dir / f"lr_vs_score_{loss_fn}_{i_p}.csv", index=False)
    fig = px.bar(df, x="abc", y="value", color="type", barmode="group")
    fig.update_xaxes(tickfont_size=6)
    fig.write_image(save_dir / f"lr_vs_score_{loss_fn}_{i_p}.png")