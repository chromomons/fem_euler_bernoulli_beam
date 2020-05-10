import numpy as np
import math as m
import matplotlib.pyplot as plt
import functools as ft
import polynomial_tools as pt

def local_stiffness_matrices(basis_functions, inner_product, master_element, NEL, gauss_points_num):
    Nb = len(basis_functions)
    Kl = np.empty([NEL,Nb,Nb], float)
    for k in range(NEL):
        for i in range(Nb):
            for j in range(Nb):
                Kl[k][i][j] = 1 / (ls[k])**3 * pt.gauss(lambda x: d2_N[i](x) * d2_N[j](x), master_element[0], master_element[1], gauss_points_num)
    return Kl
