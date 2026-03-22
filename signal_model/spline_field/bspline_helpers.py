from sympy.abc import x, y
from sympy import lambdify
from sympy.functions.special import bsplines
from itertools import product


def lambdify_bspline(bspline):
    """
    lambdifies b-splines, i.e., making them a callable function with arguments (for speed purposes)
    :param bspline: sympy function
    :return: f, callable function
    """
    if len(bspline.free_symbols) == 2:
        f = lambdify([x, y], bspline)
    elif bspline.free_symbols == set([x]):
        f = lambdify(x, bspline)
    elif bspline.free_symbols == set([y]):
        f = lambdify(y, bspline)
    else:
        raise ValueError('lambdify_bspline: variable not found')
    return f


def create_bsplines_patch_set(x_set, y_set):
    """
    Takes two 1D bsplines sets and creates 2D b-spline patches
    :param x_set: list, first 1D set
    :param y_set: list, second 1D set
    :return: patch_set, cartesian product of x_set, y_set
    """
    patch_set = [e[0] * e[1] for e in product(x_set, y_set)]
    return patch_set


def create_bsplines_set(var, d, knots):
    """
    Creates 1D bsplines set over a certain variable (var), having d order and knots vector
    :param var: sympy variable (e.g from sympy.abc import x)
    :param d: B-splines degree
    :param knots: list, knots vector
    :return: list, sympy functions of bsplines basis set
    """
    knots = list(knots)
    bsplines_set = bsplines.bspline_basis_set(d, knots, var)
    return bsplines_set


