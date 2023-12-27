from itertools import product

import numpy as np
import pandas as pd
import plotly.express as px
import torch


def get_vector_support(d):
    """
    Generate all possible values for a binary vector with dimension d.
    """
    binary_combinations = product([0, 1], repeat=d)
    return [np.array(c) for c in binary_combinations]


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


def likelihood_ratios(d):
    """
    Calculate the true likelihood ratio p(a,b,c)/p(a)p(b)p(c) = p(c|a,b)/p(c)
    for all possible values of a, b, c and all possible values of i_p.
    """
    data = {}
    for i_p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        data[i_p] = {"abc": [], "value": []}

        A = get_vector_support(d)
        B = A.copy()
        C = A.copy()
        for a in A:
            for b in B:
                for c in C:
                    p_c_given_a_b = prob_c_given_a_b(a, b, c, i_p)
                    p_c = prob_c(c, d, i_p)
                    lr = p_c_given_a_b / p_c if p_c != 0 else 0
                    data[i_p]["abc"].append(
                        "".join(np.concatenate((a, b, c)).astype(str))
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
    elif loss_fn == "clip":
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


########################
# INFORMATION MEASURES #
########################


def prob_i(i, i_p):
    return (i_p**i) * ((1-i_p)**(1-i))


def c_definition(a, b, i):
    """
    Returns c = (a XOR b)^i * a^(1-i)
    """
    if i == 1:
        return np.logical_xor(a, b).astype(int)
    elif i == 0:
        return a


def indicator_c(a, b, c, i):
    return (c == c_definition(a, b, i)).astype(int)


def prob_c_given_a(a, c, d, i_p):
    """
    Computes p(c|a) = (0.5)^d * sum_{b_1,...,b_d,i} p(i)
        \prod_{j=1}^d ind[ c_j = (a_j XOR b_j)^i * a_j^(1-i) ]
    """
    B = get_vector_support(d)
    I = [0, 1]

    sum = 0
    for b in B:
        for i in I:
            ind_c = indicator_c(a, b, c, i)
            prod = np.prod(ind_c)
            p_i = prob_i(i, i_p)
            sum += p_i * prod
    return (0.5)**d * sum


def prob_c_given_b(b, c, d, i_p):
    """
    Computes p(c|b) = (0.5)^d * sum_{a_1,...,a_d,i} p(i)
        \prod_{j=1}^d ind[ c_j = (a_j XOR b_j)^i * a_j^(1-i) ]
    """
    A = get_vector_support(d)
    I = [0, 1]

    sum = 0
    for a in A:
        for i in I:
            ind_c = indicator_c(a, b, c, i)
            prod = np.prod(ind_c)
            p_i = prob_i(i, i_p)
            sum += p_i * prod
    return (0.5)**d * sum


def prob_c(c, d, i_p):
    """
    Computes p(c) = (0.5)^{2d} * sum_{a_1,...,a_d,b_1,...,b_d,i} p(i)
        \prod_{j=1}^d ind[ c_j = (a_j XOR b_j)^i * a_j^(1-i) ]
    """
    A = get_vector_support(d)
    B = A.copy()
    I = [0, 1]
    sum = 0
    for a in A:
        for b in B:
            for i in I:
                ind_c = indicator_c(a, b, c, i)
                prod = np.prod(ind_c)
                p_i = prob_i(i, i_p)
                sum += p_i * prod
    return (0.5)**(2*d) * sum


def prob_c_given_a_b(a, b, c, i_p):
    """
    Computes p(c|a,b) = sum_{i} p(i)
        \prod_{j=1}^d ind[ c_j = (a_j XOR b_j)^i * a_j^(1-i) ]
    """
    I = [0, 1]
    sum = 0
    for i in I:
        ind_c = indicator_c(a, b, c, i)
        prod = np.prod(ind_c)
        p_i = prob_i(i, i_p)
        sum += p_i * prod
    return sum


def prob_a_given_c(a, c, d, i_p):
    """
    Computes p(a|c) = (0.5)^d * p(c|a) / p(c)
    """
    p_c_given_a = prob_c_given_a(a, c, d, i_p)
    p_c = prob_c(c, d, i_p)
    return (0.5)**d * (p_c_given_a / p_c)


def prob_a_given_c_b(a, b, c, d, i_p):
    """
    Computes p(a|c,b) = (0.5)^d * p(c|a,b) / p(c|b)
    """
    p_c_given_a_b = prob_c_given_a_b(a, b, c, i_p)
    p_c_given_b = prob_c_given_b(b, c, d, i_p)
    return (0.5)**d * (p_c_given_a_b / p_c_given_b)


def MI_a_c(d, i_p):
    """
    Computes mutual information between a and c:
    MI(a;c) = (0.5)^d * sum_{a,c} p(c|a) log[p(c|a)/p(c)]
    """
    A = get_vector_support(d)
    C = A.copy()
    sum = 0
    for a in A:
        for c in C:
            p_c_given_a = prob_c_given_a(a, c, d, i_p)
            p_c = prob_c(c, d, i_p)
            if p_c_given_a != 0:
                sum += p_c_given_a * np.log(p_c_given_a/p_c)
    return (0.5)**d * sum


def MI_b_c(d, i_p):
    """
    Computes mutual information between b and c:
    MI(b;c) = (0.5)^d * sum_{b,c} p(c|a) log[p(c|b)/p(c)]
    """
    B = get_vector_support(d)
    C = B.copy()
    sum = 0
    for b in B:
        for c in C:
            p_c_given_b = prob_c_given_b(b, c, d, i_p)
            p_c = prob_c(c, d, i_p)
            if p_c_given_b != 0:
                sum += p_c_given_b * np.log(p_c_given_b/p_c)
    return (0.5)**d * sum


def MI_a_b_given_c(d, i_p):
    """
    Computes mutual information between a and b given c:
    MI(a;b|c) = (0.5)^{2d} * sum_{a,b,c} p(c|a,b)
        log[ [p(c|a,b) * p(c)] / [p(c|a) * p(c|b)] ]
    """
    A = get_vector_support(d)
    B = A.copy()
    C = A.copy()
    sum = 0
    for a in A:
        for b in B:
            for c in C:
                p_c_given_a_b = prob_c_given_a_b(a, b, c, i_p)
                p_c_given_a = prob_c_given_a(a, c, d, i_p)
                p_c_given_b = prob_c_given_b(b, c, d, i_p)
                p_c = prob_c(c, d, i_p)
                if p_c_given_a_b != 0:
                    sum += p_c_given_a_b * np.log(
                            (p_c_given_a_b * p_c) / (p_c_given_a * p_c_given_b)
                        )
    return (0.5)**(2*d) * sum


def MI_c_b_given_a(d, i_p):
    """
    Computes mutual information between c and b given a:
    MI(c;b|a) = (0.5)^{2d} * sum_{a,b,c} p(c|a,b)
        log[ [p(a|c,b) * p(c|b)] / [p(a|c) * p(c)] ]
    """
    A = get_vector_support(d)
    B = A.copy()
    C = A.copy()
    sum = 0
    for a in A:
        for b in B:
            for c in C:
                p_c_given_a_b = prob_c_given_a_b(a, b, c, i_p)
                p_a_given_c_b = prob_a_given_c_b(a, b, c, d, i_p)
                p_a_given_c = prob_a_given_c(a, c, d, i_p)
                p_c_given_b = prob_c_given_b(b, c, d, i_p)
                p_c = prob_c(c, d, i_p)
                if p_c_given_a_b != 0:
                    sum += p_c_given_a_b * np.log(
                            (p_a_given_c_b * p_c_given_b) / (p_a_given_c * p_c)
                        )
    return (0.5)**(2*d) * sum


def mutual_informations(d, i_p):
    mi_a_c = MI_a_c(d, i_p)
    mi_b_c = MI_b_c(d, i_p)
    mi_a_b_given_c = MI_a_b_given_c(d, i_p)
    mi_c_b_given_a = MI_c_b_given_a(d, i_p)
    return {"mi_a_c": mi_a_c, "mi_b_c": mi_b_c,
            "mi_a_b_given_c": mi_a_b_given_c, "mi_c_b_given_a": mi_c_b_given_a}