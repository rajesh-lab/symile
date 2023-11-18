"""
Computes mutual informations, total correlation, and best predictor on the
binary data.

The functions below handle either the one-dimensional or two-dimensional case:

One-dimensional case:
    a, b ~ Bernoulli(0.5)
    i ~ Bernoulli(i_p)
    c = (a XOR b)^i * a^(1-i)

Two-dimensional case:
    a_1, a_2, b_1, b_2 ~ Bernoulli(0.5)
    i ~ Bernoulli(i_p)
    c_1 = (a_1 XOR b_1)^i * a_1^(1-i)
    c_2 = (a_2 XOR b_2)^i * b_1^(1-i)
"""
import numpy as np


def c_1_definition(a, b, i):
    """
    Returns c = (a XOR b)^i * a^(1-i)
    """
    return (int(np.logical_xor(a, b))**i) * (a**(1-i))


def c_2_definition(a_2, b_1, b_2, i):
    """
    Returns c_2 = (a_2 XOR b_2)^i * b_1^(1-i)
    """
    return (int(np.logical_xor(a_2, b_2))**i) * (b_1**(1-i))


def indicator_c_1(a_1, b_1, c_1, i):
    if c_1 == c_1_definition(a_1, b_1, i):
        return 1
    else:
        return 0


def indicator_c_2(a_2, b_1, b_2, c_2, i):
    if c_2 == c_2_definition(a_2, b_1, b_2, i):
        return 1
    else:
        return 0


def mutual_informations(i_p, dim):
    if dim == 1:
        mi_a_c = MI_a_c_1d(i_p)
        mi_b_c = MI_b_c_1d(i_p)
        mi_a_b_given_c = MI_a_b_given_c_1d(i_p)
    elif dim == 2:
        mi_a_c = MI_a_c_2d(i_p)
        mi_b_c = MI_b_c_2d(i_p)
        mi_a_b_given_c = MI_a_b_given_c_2d(i_p)
    return {"mi_a_c": mi_a_c, "mi_b_c": mi_b_c, "mi_a_b_given_c": mi_a_b_given_c}


def best_accuracy(i_p, dim):
    if dim == 1:
        return best_accuracy_pred_b_1d(i_p)
    elif dim == 2:
        return best_accuracy_pred_b_2d(i_p)


########################
# ONE-DIMENSIONAL CASE #
########################


def prob_c_1d(c, i_p):
    """
    Computes p(c) in the one-dimensional case:
    p(c) = 0.25 * sum_{a,b,i} [ ind[c = (a XOR b)^i * a^(1-i)] p(i) ]
    Note that this function should always return 0.5.
    """
    A = [0, 1]
    B = [0, 1]
    I = [0, 1]
    sum_a_b_i = 0
    for a in A:
        for b in B:
            for i in I:
                ind_c = indicator_c_1(a, b, c, i)
                p_of_i = (i_p**i) * ((1-i_p)**(1-i))
                sum_a_b_i += ind_c * p_of_i
    assert round(sum_a_b_i) == 2, "sum_a_b_i should always be 2."
    return 0.25 * sum_a_b_i


def prob_c_given_a_1d(a, c, i_p):
    """
    Computes p(c|a) in the one-dimensional case:
    p(c|a) = 0.5 * sum_{b,i} [ ind[c = (a XOR b)^i * a^(1-i)] p(i) ]
    """
    B = [0, 1]
    I = [0, 1]

    sum_b_i = 0
    for b in B:
        for i in I:
            ind_c = indicator_c_1(a, b, c, i)
            p_of_i = (i_p**i) * ((1-i_p)**(1-i))
            sum_b_i += ind_c * p_of_i
    return 0.5 * sum_b_i


def prob_c_given_b_1d(b, c, i_p):
    """
    Computes p(c|b) in the one-dimensional case:
    p(c|b) = 0.5 * sum_{a,i} [ ind[c = (a XOR b)^i * a^(1-i)] p(i) ]
    """
    A = [0, 1]
    I = [0, 1]
    sum_a_i = 0
    for a in A:
        for i in I:
            ind_c = indicator_c_1(a, b, c, i)
            p_of_i = (i_p**i) * ((1-i_p)**(1-i))
            sum_a_i += ind_c * p_of_i
    return 0.5 * sum_a_i


def prob_c_given_a_b_1d(a, b, c, i_p):
    """
    Computes p(c|a,b) in the one-dimensional case:
    p(c|a,b) = sum_i [ ind[c = (a XOR b)^i * a^(1-i)] p(i) ]
    """
    I = [0, 1]
    sum_i = 0
    for i in I:
        ind_c = indicator_c_1(a, b, c, i)
        p_of_i = (i_p**i) * ((1-i_p)**(1-i))
        sum_i += ind_c * p_of_i
    return sum_i


def MI_a_c_1d(i_p):
    """
    Computes mutual information between a and c in the one-dimensional case:
    MI(a;c) = 0.5 * sum_{a,c} [ p(c|a) log[p(c|a)/p(c)] ]
    """
    A = [0, 1]
    C = [0, 1]
    sum_a_c = 0
    for a in A:
        for c in C:
            p_c_given_a = prob_c_given_a_1d(a, c, i_p)
            p_c = prob_c_1d(c, i_p)
            if p_c_given_a != 0:
                sum_a_c += p_c_given_a * np.log(p_c_given_a/p_c)
    return 0.5 * sum_a_c


def MI_b_c_1d(i_p):
    """
    Computes mutual information between b and c in the one-dimensional case:
    MI(b;c) = 0.5 * sum_{b,c} [ p(c|b) log[p(c|b)/p(c)] ]
    """
    B = [0, 1]
    C = [0, 1]
    sum_b_c = 0
    for b in B:
        for c in C:
            p_c_given_b = prob_c_given_b_1d(b, c, i_p)
            p_c = prob_c_1d(c, i_p)
            if p_c_given_b != 0:
                sum_b_c += p_c_given_b * np.log(p_c_given_b/p_c)
    return 0.5 * sum_b_c


def MI_a_b_given_c_1d(i_p):
    """
    Computes mutual information between a and b given c in the one-dimensional case:
    MI(a;b|c) = 0.25 * sum_{a,b,c} [ p(c|a,b) log[(p(c|a,b)p(c))/(p(c|a)p(c|b))] ]
    """
    A = [0, 1]
    B = [0, 1]
    C = [0, 1]
    sum_a_b_c = 0
    for a in A:
        for b in B:
            for c in C:
                p_c_given_a_b = prob_c_given_a_b_1d(a, b, c, i_p)
                p_c_given_a = prob_c_given_a_1d(a, c, i_p)
                p_c_given_b = prob_c_given_b_1d(b, c, i_p)
                p_c = prob_c_1d(c, i_p)
                if p_c_given_a_b != 0:
                    sum_a_b_c += p_c_given_a_b * np.log(
                            (p_c_given_a_b * p_c) / (p_c_given_a * p_c_given_b)
                        )
    return 0.25 * sum_a_b_c


def prob_b_given_a_c_1d(a, b, c, i_p):
    """
    Computes p(b|a,c) in the one-dimensional case:
    p(b|a,c) = p(c|a,b) / sum_{b'} p(c|a,b')
    """
    p_c_given_a_b = prob_c_given_a_b_1d(a, b, c, i_p)

    B = [0, 1]
    sum_b = 0
    for b_prime in B:
        sum_b += prob_c_given_a_b_1d(a, b_prime, c, i_p)
    if p_c_given_a_b == 0:
        return 0
    else:
        return p_c_given_a_b/sum_b


def best_predictor_b_1d(a, c, i_p):
    """Returns the b that maximizes p(b|a,c) in the one-dimensional case."""
    B = [0, 1]
    best_p_b_given_a_c = -np.inf
    b_pred = None
    for b in B:
        p_b_given_a_c = prob_b_given_a_c_1d(a, b, c, i_p)
        if p_b_given_a_c > best_p_b_given_a_c:
            best_p_b_given_a_c = p_b_given_a_c
            b_pred = b
    return b_pred


def best_accuracy_pred_b_1d(i_p):
    """
    Computes the best possible accuracy when predicting b given a, c
    in the one-dimensional case:
    p(a) * p(b) * sum_{a, b, c} p(c|a,b) * ind[b = b_pred]
    """
    A = [0, 1]
    B = [0, 1]
    C = [0, 1]
    sum_a_b_c = 0
    for a in A:
        for b in B:
            for c in C:
                p_c_given_a_b = prob_c_given_a_b_1d(a, b, c, i_p)
                b_pred = best_predictor_b_1d(a, c, i_p)
                if b_pred == b:
                    accurate_prediction = 1
                else:
                    accurate_prediction = 0
                sum_a_b_c += p_c_given_a_b * accurate_prediction
    return 0.25 * sum_a_b_c


########################
# TWO-DIMENSIONAL CASE #
########################


def prob_c_given_a_2d(a_1, a_2, c_1, c_2, i_p):
    """
    Computes p(c|a) in the two-dimensional case:
    p(c|a) = 0.25 * sum_{b_1,b_2,i} [ ind[c_1 = (a_1 XOR b_1)^i * a_1^(1-i)]
                                      ind[c_2 = (a_2 XOR b_2)^i * b_1^(1-i)] p(i) ]
    """
    B_1 = [0, 1]
    B_2 = [0, 1]
    I = [0, 1]
    sum_b_i = 0
    for b_1 in B_1:
        for b_2 in B_2:
            for i in I:
                ind_c_1 = indicator_c_1(a_1, b_1, c_1, i)
                ind_c_2 = indicator_c_2(a_2, b_1, b_2, c_2, i)
                p_of_i = (i_p**i) * ((1-i_p)**(1-i))
                sum_b_i += ind_c_1 * ind_c_2 * p_of_i
    return 0.25 * sum_b_i


def prob_c_given_b_2d(b_1, b_2, c_1, c_2, i_p):
    """
    Computes p(c|b) in the two-dimensional case:
    p(c|b) = 0.25 * sum_{a_1,a_2,i} [ ind[c_1 = (a_1 XOR b_1)^i * a_1^(1-i)]
                                      ind[c_2 = (a_2 XOR b_2)^i * b_1^(1-i)] p(i) ]
    """
    A_1 = [0, 1]
    A_2 = [0, 1]
    I = [0, 1]

    sum_a_i = 0
    for a_1 in A_1:
        for a_2 in A_2:
            for i in I:
                ind_c_1 = indicator_c_1(a_1, b_1, c_1, i)
                ind_c_2 = indicator_c_2(a_2, b_1, b_2, c_2, i)
                p_of_i = (i_p**i) * ((1-i_p)**(1-i))
                sum_a_i += ind_c_1 * ind_c_2 * p_of_i
    return 0.25 * sum_a_i


def prob_c_2d(c_1, c_2, i_p):
    """
    Computes p(c) in the two-dimensional case:
    p(c) = 0.0625 * sum_{a_1,a_2,b_1,b_2,i} [ ind[c_1 = (a_1 XOR b_1)^i * a_1^(1-i)]
                                              ind[c_2 = (a_2 XOR b_2)^i * b_1^(1-i)] p(i) ]
    Note that this function should always return 0.25.
    """
    A_1 = [0, 1]
    A_2 = [0, 1]
    B_1 = [0, 1]
    B_2 = [0, 1]
    I = [0, 1]
    sum_a_b_i = 0
    for a_1 in A_1:
        for a_2 in A_2:
            for b_1 in B_1:
                for b_2 in B_2:
                    for i in I:
                        ind_c_1 = indicator_c_1(a_1, b_1, c_1, i)
                        ind_c_2 = indicator_c_2(a_2, b_1, b_2, c_2, i)
                        p_of_i = (i_p**i) * ((1-i_p)**(1-i))
                        sum_a_b_i += ind_c_1 * ind_c_2 * p_of_i
    assert round(sum_a_b_i) == 4, "sum_a_b_i should always be 4."
    return 0.0625 * sum_a_b_i


def prob_c_given_a_b_2d(a_1, a_2, b_1, b_2, c_1, c_2, i_p):
    """
    Computes p(c|a,b) in the two-dimensional case:
    p(c|a,b) = sum_{i} [ ind[c_1 = (a_1 XOR b_1)^i * a_1^(1-i)]
                         ind[c_2 = (a_2 XOR b_2)^i * b_1^(1-i)] p(i) ]
    """
    I = [0, 1]
    sum_i = 0
    for i in I:
        ind_c_1 = indicator_c_1(a_1, b_1, c_1, i)
        ind_c_2 = indicator_c_2(a_2, b_1, b_2, c_2, i)
        p_of_i = (i_p**i) * ((1-i_p)**(1-i))
        sum_i += ind_c_1 * ind_c_2 * p_of_i
    return sum_i

def MI_a_c_2d(i_p):
    """
    Computes mutual information between a and c in the two-dimensional case:
    MI(a_1,a_2;c_1,c_2) = 0.25 * sum_{a_1,a_2,c_1,c_2} [
                        p(c_1,c_2|a_1,a_2) log[p(c_1,c_2|a_1,a_2)/p(c_1,c_2)]
                    ]
    """
    A_1 = [0, 1]
    A_2 = [0, 1]
    C_1 = [0, 1]
    C_2 = [0, 1]
    sum_a_c = 0
    for a_1 in A_1:
        for a_2 in A_2:
            for c_1 in C_1:
                for c_2 in C_2:
                    p_c_given_a = prob_c_given_a_2d(a_1, a_2, c_1, c_2, i_p)
                    p_c = prob_c_2d(c_1, c_2, i_p)
                    if p_c_given_a != 0:
                        sum_a_c += p_c_given_a * np.log(p_c_given_a/p_c)
    return 0.25 * sum_a_c


def MI_b_c_2d(i_p):
    """
    Computes mutual information between b and c in the two-dimensional case:
    MI(b_1,b_2;c_1,c_2) = 0.25 * sum_{b_1,b_2,c_1,c_2} [
                        p(c_1,c_2|b_1,b_2) log[p(c_1,c_2|b_1,b_2)/p(c_1,c_2)]
                    ]
    """
    B_1 = [0, 1]
    B_2 = [0, 1]
    C_1 = [0, 1]
    C_2 = [0, 1]
    sum_b_c = 0
    for b_1 in B_1:
        for b_2 in B_2:
            for c_1 in C_1:
                for c_2 in C_2:
                    p_c_given_b = prob_c_given_b_2d(b_1, b_2, c_1, c_2, i_p)
                    p_c = prob_c_2d(c_1, c_2, i_p)
                    if p_c_given_b != 0:
                        sum_b_c += p_c_given_b * np.log(p_c_given_b/p_c)
    return 0.25 * sum_b_c


def MI_a_b_given_c_2d(i_p):
    """
    Computes mutual information between a and b given c in the two-dimensional case:
    MI(a_1,a_2;b_1,b_2|c_1,c_2) = 0.0625 * sum_{a_1,a_2,b_1,b_2,c_1,c_2} [
        p(c_1,c_2|a_1,a_2,b_1,b_2)
        * log[(p(c_1,c_2|a_1,a_2,b_1,b_2)p(c_1,c_2))/(p(c_1,c_2|a_1,a_2)p(c_1,c_2|b_1,b_2))]
    ]
    """
    A_1 = [0, 1]
    A_2 = [0, 1]
    B_1 = [0, 1]
    B_2 = [0, 1]
    C_1 = [0, 1]
    C_2 = [0, 1]
    sum_a_b_c = 0
    for a_1 in A_1:
        for a_2 in A_2:
            for b_1 in B_1:
                for b_2 in B_2:
                    for c_1 in C_1:
                        for c_2 in C_2:
                            p_c_given_a_b = prob_c_given_a_b_2d(a_1, a_2, b_1, b_2,
                                                                c_1, c_2, i_p)
                            p_c_given_a = prob_c_given_a_2d(a_1, a_2, c_1, c_2, i_p)
                            p_c_given_b = prob_c_given_b_2d(b_1, b_2, c_1, c_2, i_p)
                            p_c = prob_c_2d(c_1, c_2, i_p)
                            if p_c_given_a_b != 0:
                                sum_a_b_c += p_c_given_a_b * np.log(
                                        (p_c_given_a_b * p_c) / (p_c_given_a * p_c_given_b)
                                    )
    return 0.0625 * sum_a_b_c


def prob_b_given_a_c_2d(a_1, a_2, b_1, b_2, c_1, c_2, i_p):
    """
    Computes p(b|a,c) in the two-dimensional case:
    p(b|a,c) = p(c|a,b) / sum_{b'} p(c|a,b')
    """
    p_c_given_a_b = prob_c_given_a_b_2d(a_1, a_2, b_1, b_2, c_1, c_2, i_p)

    B_1 = [0, 1]
    B_2 = [0, 1]
    sum_b = 0
    for b_1_prime in B_1:
        for b_2_prime in B_2:
            sum_b += prob_c_given_a_b_2d(a_1, a_2, b_1_prime, b_2_prime, c_1, c_2, i_p)
    if p_c_given_a_b == 0:
        return 0
    else:
        return p_c_given_a_b/sum_b


def best_predictor_b_2d(a_1, a_2, c_1, c_2, i_p):
    """Returns the b_1, b_2 that maximize p(b_1, b_2|a_1, a_2, c_1, c_2)."""
    B_1 = [0, 1]
    B_2 = [0, 1]
    best_p_b_given_a_c = -np.inf
    b_1_pred = None
    b_2_pred = None
    for b_1 in B_1:
        for b_2 in B_2:
            p_b_given_a_c = prob_b_given_a_c_2d(a_1, a_2, b_1, b_2, c_1, c_2, i_p)
            if p_b_given_a_c > best_p_b_given_a_c:
                best_p_b_given_a_c = p_b_given_a_c
                b_1_pred = b_1
                b_2_pred = b_2
    return b_1_pred, b_2_pred


def best_accuracy_pred_b_2d(i_p):
    """
    Computes the best possible accuracy when predicting b given a, c
    in the two-dimensional case:
    p(a_1) * p(a_2) * p(b_1) * p(b_2) * sum_{a_1, a_2, b_1, b_2, c_1, c_2} [
        p(c_1, c_2|a_1, a_2, b_1, b_2) * ind[b_1, b_2 = b_1_pred, b_2_pred]
    ]
    """
    A_1 = [0, 1]
    A_2 = [0, 1]
    B_1 = [0, 1]
    B_2 = [0, 1]
    C_1 = [0, 1]
    C_2 = [0, 1]

    sum_a_b_c = 0
    for a_1 in A_1:
        for a_2 in A_2:
            for b_1 in B_1:
                for b_2 in B_2:
                    for c_1 in C_1:
                        for c_2 in C_2:
                            p_c_given_a_b = prob_c_given_a_b_2d(a_1, a_2, b_1, b_2,
                                                                c_1, c_2, i_p)
                            b_1_pred, b_2_pred = best_predictor_b_2d(a_1, a_2,
                                                                     c_1, c_2, i_p)
                            if b_1_pred == b_1 and b_2_pred == b_2:
                                accurate_prediction = 1
                            else:
                                accurate_prediction = 0
                            sum_a_b_c += p_c_given_a_b * accurate_prediction
    return 0.0625 * sum_a_b_c