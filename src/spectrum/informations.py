import numpy as np


def c_1_definition(a_1, b_1, i):
    return (int(np.logical_xor(a_1, b_1))**i) * (a_1**(1-i))


def c_2_definition(a_2, b_1, b_2, i):
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

def prob_c_given_a(a_1, a_2, c_1, c_2, i_p):
    B_1 = [0, 1]
    B_2 = [0, 1]
    I = [0, 1]

    sum = 0
    for b_1 in B_1:
        for b_2 in B_2:
            for i in I:
                ind_c_1 = indicator_c_1(a_1, b_1, c_1, i)
                ind_c_2 = indicator_c_2(a_2, b_1, b_2, c_2, i)
                p_of_i = (i_p**i) * ((1-i_p)**(1-i))
                sum += ind_c_1 * ind_c_2 * p_of_i
    return 0.25 * sum


def prob_c_given_b(b_1, b_2, c_1, c_2, i_p):
    A_1 = [0, 1]
    A_2 = [0, 1]
    I = [0, 1]

    sum = 0
    for a_1 in A_1:
        for a_2 in A_2:
            for i in I:
                ind_c_1 = indicator_c_1(a_1, b_1, c_1, i)
                ind_c_2 = indicator_c_2(a_2, b_1, b_2, c_2, i)
                p_of_i = (i_p**i) * ((1-i_p)**(1-i))
                sum += ind_c_1 * ind_c_2 * p_of_i
    return 0.25 * sum


def prob_c(c_1, c_2, i_p):
    A_1 = [0, 1]
    A_2 = [0, 1]
    B_1 = [0, 1]
    B_2 = [0, 1]
    I = [0, 1]

    sum = 0
    for a_1 in A_1:
        for a_2 in A_2:
            for b_1 in B_1:
                for b_2 in B_2:
                    for i in I:
                        ind_c_1 = indicator_c_1(a_1, b_1, c_1, i)
                        ind_c_2 = indicator_c_2(a_2, b_1, b_2, c_2, i)
                        p_of_i = (i_p**i) * ((1-i_p)**(1-i))
                        sum += ind_c_1 * ind_c_2 * p_of_i
    return 0.0625 * sum


def prob_c_given_a_b(a_1, a_2, b_1, b_2, c_1, c_2, i_p):
    I = [0, 1]

    sum = 0
    for i in I:
        ind_c_1 = indicator_c_1(a_1, b_1, c_1, i)
        ind_c_2 = indicator_c_2(a_2, b_1, b_2, c_2, i)
        p_of_i = (i_p**i) * ((1-i_p)**(1-i))
        sum += ind_c_1 * ind_c_2 * p_of_i
    return sum


def MI_a_c(i_p):
    A_1 = [0, 1]
    A_2 = [0, 1]
    C_1 = [0, 1]
    C_2 = [0, 1]

    sum = 0
    for a_1 in A_1:
        for a_2 in A_2:
            for c_1 in C_1:
                for c_2 in C_2:
                    p_c_given_a = prob_c_given_a(a_1, a_2, c_1, c_2, i_p)
                    p_c = prob_c(c_1, c_2, i_p)
                    if p_c_given_a != 0:
                        sum += p_c_given_a * np.log(p_c_given_a/p_c)
    return 0.25 * sum


def MI_b_c(i_p):
    B_1 = [0, 1]
    B_2 = [0, 1]
    C_1 = [0, 1]
    C_2 = [0, 1]

    sum = 0
    for b_1 in B_1:
        for b_2 in B_2:
            for c_1 in C_1:
                for c_2 in C_2:
                    p_c_given_b = prob_c_given_b(b_1, b_2, c_1, c_2, i_p)
                    p_c = prob_c(c_1, c_2, i_p)
                    if p_c_given_b != 0:
                        sum += p_c_given_b * np.log(p_c_given_b/p_c)
    return 0.25 * sum


def MI_a_b_given_c(i_p):
    A_1 = [0, 1]
    A_2 = [0, 1]
    B_1 = [0, 1]
    B_2 = [0, 1]
    C_1 = [0, 1]
    C_2 = [0, 1]

    sum = 0
    for a_1 in A_1:
        for a_2 in A_2:
            for b_1 in B_1:
                for b_2 in B_2:
                    for c_1 in C_1:
                        for c_2 in C_2:
                            p_c_given_a_b = prob_c_given_a_b(a_1, a_2, b_1, b_2,
                                                             c_1, c_2, i_p)
                            p_c_given_a = prob_c_given_a(a_1, a_2, c_1, c_2, i_p)
                            p_c_given_b = prob_c_given_b(b_1, b_2, c_1, c_2, i_p)
                            p_c = prob_c(c_1, c_2, i_p)
                            if p_c_given_a_b != 0:
                                sum += p_c_given_a_b * np.log(
                                        (p_c_given_a_b * p_c) / (p_c_given_a * p_c_given_b)
                                    )
    return 0.0625 * sum

def mutual_informations(i_p):
    mi_a_c = MI_a_c(i_p)
    mi_b_c = MI_b_c(i_p)
    mi_a_b_given_c = MI_a_b_given_c(i_p)
    return {"mi_a_c": mi_a_c, "mi_b_c": mi_b_c, "mi_a_b_given_c": mi_a_b_given_c}


def prob_a_given_b_c(a_1, a_2, b_1, b_2, c_1, c_2, i_p):
    I = [0, 1]

    sum = 0
    for i in I:
        ind_c_1 = indicator_c_1(a_1, b_1, c_1, i)
        ind_c_2 = indicator_c_2(a_2, b_1, b_2, c_2, i)
        p_of_i = (i_p**i) * ((1-i_p)**(1-i))
        sum += ind_c_1 * ind_c_2 * p_of_i
    return sum


def best_predictor(b_1, b_2, c_1, c_2, i_p):
    A_1 = [0, 1]
    A_2 = [0, 1]

    best_p_a_given_b_c = -np.inf
    best_a_1 = None
    best_a_2 = None
    for a_1 in A_1:
        for a_2 in A_2:
            p_a_given_b_c = prob_a_given_b_c(a_1, a_2, b_1, b_2, c_1, c_2, i_p)
            if p_a_given_b_c > best_p_a_given_b_c:
                best_p_a_given_b_c = p_a_given_b_c
                best_a_1 = a_1
                best_a_2 = a_2
    return best_a_1, best_a_2


def best_accuracy(i_p):
    A_1 = [0, 1]
    A_2 = [0, 1]
    B_1 = [0, 1]
    B_2 = [0, 1]
    C_1 = [0, 1]
    C_2 = [0, 1]

    sum = 0
    for a_1 in A_1:
        for a_2 in A_2:
            for b_1 in B_1:
                for b_2 in B_2:
                    for c_1 in C_1:
                        for c_2 in C_2:
                            p_c_given_a_b = prob_c_given_a_b(a_1, a_2, b_1, b_2,
                                                             c_1, c_2, i_p)
                            best_a_1, best_a_2 = best_predictor(b_1, b_2, c_1, c_2, i_p)
                            if best_a_1 == a_1 and best_a_2 == a_2:
                                accurate_prediction = 1
                            else:
                                accurate_prediction = 0
                            sum += p_c_given_a_b * accurate_prediction
    return 0.0625 * sum