import numpy as np
import functools as ft

def get_monomial_basis(size):
    return np.array([(lambda pow: lambda x: x**pow)(pow) for pow in range(size)])

def get_basis_functions(coefficient_matrix):
    size = len(coefficient_matrix)
    monomial_basis = get_monomial_basis(size)
    basis = [ft.reduce( (lambda monom1, monom2: lambda x: monom1(x) + monom2(x)), [(lambda monom, scalar: lambda x: monom(x)*scalar)(monomial_basis[j], coefficient_matrix[i,j]) for j in range(size)]) for i in range(size)]
    return np.array(basis).ravel()

def get_basis_derivatives(coefficient_matrix, order):
    der_mat = get_derivative_matrix(len(coefficient_matrix), order)
    return get_basis_functions(coefficient_matrix@der_mat)

def get_derivative_matrix(basis_order, derivative_order):
    derivative_matrix = np.zeros((basis_order, basis_order))
    for i in range(derivative_order, basis_order):
        derivative_matrix[i,i-derivative_order] = ft.reduce(lambda xx,yy: xx*yy, range(i,i-derivative_order,-1))
    return derivative_matrix

def gauss(fun, a, b, gauss_points_num):
    x, w = np.polynomial.legendre.leggauss(gauss_points_num)
    return ft.reduce(lambda xx,yy: xx+yy, map(lambda xx,ww: (b-a)/2*ww*fun((b-a)/2*xx + (a+b)/2), x,w))
