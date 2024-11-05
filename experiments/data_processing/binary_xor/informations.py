"""
This script calculates I(a,c), I(b,c), I(a;b|c), I(c;b|a), and TC(a,b,c)
for each p_hat in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}.
"""
from datetime import datetime
import os
import time

import numpy as np
import pandas as pd

from args import parse_args_informations
from utils import get_vector_support


def arr_as_str(vector):
    return "".join(map(str, vector))


def prob_i(i, p_hat):
    return (p_hat**i) * ((1-p_hat)**(1-i))


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


def prob_c_given_a(a, c, d, p_hat):
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
            p_i = prob_i(i, p_hat)
            sum += p_i * prod
    return (0.5)**d * sum


def prob_c_given_a_df(d, p_hat):
    A = get_vector_support(d)
    C = A.copy()
    p_c_given_a_dict = {"a": [], "c": [], "p_c_given_a": []}
    for a in A:
        for c in C:
            p_c_given_a = prob_c_given_a(a, c, d, p_hat)

            p_c_given_a_dict["a"].append(arr_as_str(a))
            p_c_given_a_dict["c"].append(arr_as_str(c))
            p_c_given_a_dict["p_c_given_a"].append(p_c_given_a)
    return pd.DataFrame(p_c_given_a_dict)


def prob_c_given_b(b, c, d, p_hat):
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
            p_i = prob_i(i, p_hat)
            sum += p_i * prod
    return (0.5)**d * sum


def prob_c_given_b_df(d, p_hat):
    B = get_vector_support(d)
    C = B.copy()
    p_c_given_b_dict = {"b": [], "c": [], "p_c_given_b": []}
    for b in B:
        for c in C:
            p_c_given_b = prob_c_given_b(b, c, d, p_hat)

            p_c_given_b_dict["b"].append(arr_as_str(b))
            p_c_given_b_dict["c"].append(arr_as_str(c))
            p_c_given_b_dict["p_c_given_b"].append(p_c_given_b)
    return pd.DataFrame(p_c_given_b_dict)


def prob_c(c, d, p_hat):
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
                p_i = prob_i(i, p_hat)
                sum += p_i * prod
    return (0.5)**(2*d) * sum


def prob_c_df(d, p_hat):
    C = get_vector_support(d)
    p_c_dict = {"c": [], "p_c": []}
    for c in C:
        p_c = prob_c(c, d, p_hat)

        p_c_dict["c"].append(arr_as_str(c))
        p_c_dict["p_c"].append(p_c)
    return pd.DataFrame(p_c_dict)


def prob_c_given_a_b(a, b, c, p_hat):
    """
    Computes p(c|a,b) = sum_{i} p(i)
        \prod_{j=1}^d ind[ c_j = (a_j XOR b_j)^i * a_j^(1-i) ]
    """
    I = [0, 1]
    sum = 0
    for i in I:
        ind_c = indicator_c(a, b, c, i)
        prod = np.prod(ind_c)
        p_i = prob_i(i, p_hat)
        sum += p_i * prod
    return sum


def prob_a_given_c_b(a, b, c, d, p_hat):
    """
    Computes p(a|c,b) = (0.5)^d * p(c|a,b) / p(c|b)
    """
    p_c_given_a_b = prob_c_given_a_b(a, b, c, p_hat)
    p_c_given_b = prob_c_given_b(b, c, d, p_hat)
    return (0.5)**d * (p_c_given_a_b / p_c_given_b)


def prob_c_given_a_b_and_prob_a_given_c_b_df(d, p_hat):
    A = get_vector_support(d)
    B = A.copy()
    C = A.copy()
    p_c_given_a_b_dict = {"a": [], "b": [], "c": [], "p_c_given_a_b": []}
    p_a_given_c_b_dict = {"a": [], "b": [], "c": [], "p_a_given_c_b": []}
    for a in A:
        for b in B:
            for c in C:
                p_c_given_a_b = prob_c_given_a_b(a, b, c, p_hat)
                p_c_given_a_b_dict["a"].append(arr_as_str(a))
                p_c_given_a_b_dict["b"].append(arr_as_str(b))
                p_c_given_a_b_dict["c"].append(arr_as_str(c))
                p_c_given_a_b_dict["p_c_given_a_b"].append(p_c_given_a_b)

                p_a_given_c_b = prob_a_given_c_b(a, b, c, d, p_hat)
                p_a_given_c_b_dict["a"].append(arr_as_str(a))
                p_a_given_c_b_dict["b"].append(arr_as_str(b))
                p_a_given_c_b_dict["c"].append(arr_as_str(c))
                p_a_given_c_b_dict["p_a_given_c_b"].append(p_a_given_c_b)
    return pd.DataFrame(p_c_given_a_b_dict), pd.DataFrame(p_a_given_c_b_dict)


def prob_a_given_c(d, p_c_given_a, p_c):
    """
    Computes p(a|c) = (0.5)^d * p(c|a) / p(c)
    """
    return (0.5)**d * (p_c_given_a / p_c)


def MI_a_c(d, p_c_given_a_df, p_c_df):
    """
    Computes mutual information between a and c:
    MI(a;c) = (0.5)^d * sum_{a,c} p(c|a) log[p(c|a)/p(c)]
    """
    A = get_vector_support(d)
    C = A.copy()
    sum = 0
    for a in A:
        for c in C:
            a = arr_as_str(a)
            c = arr_as_str(c)

            p_c_given_a = p_c_given_a_df[
                    (p_c_given_a_df["a"]==a) & (p_c_given_a_df["c"]==c)
                ].p_c_given_a.item()

            p_c = p_c_df[p_c_df["c"]==c].p_c.item()

            if p_c_given_a != 0:
                sum += p_c_given_a * np.log(p_c_given_a/p_c)

    return (0.5)**d * sum


def MI_b_c(d, p_c_given_b_df, p_c_df):
    """
    Computes mutual information between b and c:
    MI(b;c) = (0.5)^d * sum_{b,c} p(c|a) log[p(c|b)/p(c)]
    """
    B = get_vector_support(d)
    C = B.copy()
    sum = 0
    for b in B:
        for c in C:
            b = arr_as_str(b)
            c = arr_as_str(c)

            p_c_given_b = p_c_given_b_df[
                    (p_c_given_b_df["b"]==b) & (p_c_given_b_df["c"]==c)
                ].p_c_given_b.item()

            p_c = p_c_df[p_c_df["c"]==c].p_c.item()

            if p_c_given_b != 0:
                sum += p_c_given_b * (
                    np.log(p_c_given_b) - np.log(p_c)
                )
    return (0.5)**d * sum


def MI_a_b_given_c(d, p_c_given_a_b_df, p_c_given_a_df, p_c_given_b_df, p_c_df):
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
                a = arr_as_str(a)
                b = arr_as_str(b)
                c = arr_as_str(c)

                p_c_given_a_b = p_c_given_a_b_df[
                    (p_c_given_a_b_df["a"]==a) & (p_c_given_a_b_df["b"]==b) & (p_c_given_a_b_df["c"]==c)
                ].p_c_given_a_b.item()

                p_c_given_a = p_c_given_a_df[
                    (p_c_given_a_df["a"]==a) & (p_c_given_a_df["c"]==c)
                ].p_c_given_a.item()

                p_c_given_b = p_c_given_b_df[
                    (p_c_given_b_df["b"]==b) & (p_c_given_b_df["c"]==c)
                ].p_c_given_b.item()

                p_c = p_c_df[p_c_df["c"]==c].p_c.item()

                if p_c_given_a_b != 0:
                    sum += p_c_given_a_b * (
                        np.log(p_c_given_a_b) + np.log(p_c) - np.log(p_c_given_a) - np.log(p_c_given_b)
                    )
    return (0.5)**(2*d) * sum


def MI_c_b_given_a(d, p_c_given_a_b_df, p_a_given_c_b_df, p_c_given_b_df, p_c_df, p_c_given_a_df):
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
                a = arr_as_str(a)
                b = arr_as_str(b)
                c = arr_as_str(c)

                p_c_given_a_b = p_c_given_a_b_df[
                    (p_c_given_a_b_df["a"]==a) & (p_c_given_a_b_df["b"]==b) & (p_c_given_a_b_df["c"]==c)
                ].p_c_given_a_b.item()

                p_c_given_b = p_c_given_b_df[
                    (p_c_given_b_df["b"]==b) & (p_c_given_b_df["c"]==c)
                ].p_c_given_b.item()

                p_c = p_c_df[p_c_df["c"]==c].p_c.item()

                p_a_given_c_b = p_a_given_c_b_df[
                    (p_a_given_c_b_df["a"]==a) & (p_a_given_c_b_df["b"]==b) & (p_a_given_c_b_df["c"]==c)
                ].p_a_given_c_b.item()

                p_c_given_a = p_c_given_a_df[
                    (p_c_given_a_df["a"]==a) & (p_c_given_a_df["c"]==c)
                ].p_c_given_a.item()

                p_a_given_c = prob_a_given_c(d, p_c_given_a, p_c)

                if p_c_given_a_b != 0:
                    sum += p_c_given_a_b * (
                        np.log(p_a_given_c_b) + np.log(p_c_given_b) - np.log(p_a_given_c) - np.log(p_c)
                    )
    return (0.5)**(2*d) * sum


def mutual_informations(d, p_hat):
    print("creating dataframes...")
    p_c_given_a_df = prob_c_given_a_df(d, p_hat)
    p_c_given_b_df = prob_c_given_b_df(d, p_hat)
    p_c_df = prob_c_df(d, p_hat)
    p_c_given_a_b_df, p_a_given_c_b_df = prob_c_given_a_b_and_prob_a_given_c_b_df(d, p_hat)

    print("computing mi_a_c...")
    mi_a_c = MI_a_c(d, p_c_given_a_df, p_c_df)

    print("computing mi_b_c...")
    mi_b_c = MI_b_c(d, p_c_given_b_df, p_c_df)

    print("computing mi_a_b_given_c...")
    mi_a_b_given_c = MI_a_b_given_c(d, p_c_given_a_b_df, p_c_given_a_df, p_c_given_b_df, p_c_df)

    print("computing mi_c_b_given_a...")
    mi_c_b_given_a = MI_c_b_given_a(d, p_c_given_a_b_df, p_a_given_c_b_df, p_c_given_b_df, p_c_df, p_c_given_a_df)

    return {"mi_a_c": mi_a_c, "mi_b_c": mi_b_c,
            "mi_a_b_given_c": mi_a_b_given_c, "mi_c_b_given_a": mi_c_b_given_a}


if __name__ == '__main__':
    start = time.time()

    args = parse_args_informations()

    datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir / datetime_now
    os.makedirs(save_dir)
    print(f"\nResults will be saved in {save_dir}.\n")

    mi_results = {"p_hat": [], "value": [], "type": []}

    for p_hat in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f"computing information terms for p_hat = {p_hat}...")

        mi = mutual_informations(args.d_v, p_hat)

        mi["total_corr"] = mi["mi_a_c"] + mi["mi_b_c"] + mi["mi_a_b_given_c"]

        for k, v in mi.items():
            mi_results["p_hat"].append(p_hat)
            mi_results["type"].append(k)
            mi_results["value"].append(v)

    mi_df = pd.DataFrame(mi_results)
    mi_df.to_csv(save_dir / "mi.csv", index=False)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")