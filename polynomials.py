import typing

import numpy as np


def solution_points(
    x1: float,
    x2: float,
    n: int,
) -> np.ndarray:
    """
    Return `n + 1` Chebyshev-like points on the [`x1`, `x2`] interval.

    They therefore lie between corresponding points of the `n`th degree
    Gauss-Legendre quadrature together with the bounds.
    """
    i = np.arange(n + 1)
    return x1 + (1.0 - np.cos((2*i + 1)/(2*(n + 1))*np.pi))*(x2 - x1)/2.0


def gauss_legendre_quadrature(
    x1: float,
    x2: float,
    n: int,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Return `n` points and weights from the `n`th degree
    Gauss-Legendre quadrature on the interval [`x1`, `x2`].
    """
    if n == 0:
        return np.ndarray(0), np.ndarray(0)
    x, w = np.polynomial.legendre.leggauss(n)
    return x1 + (x + 1.0)*(x2 - x1)/2.0, w*(x2 - x1)/2.0

def flux_points( 
    x1: float,
    x2: float,
    p: int,
)->np.ndarray:
    """
    Return `p+2` points from the `p`th degree
    Gauss-Legendre quadrature on the interval [`x1`, `x2`].
    """
    x_fp = gauss_legendre_quadrature(0,1,p)[0]
    return np.hstack((0.0, x_fp, 1.0))


def lagrange_matrix(
    x_to: np.ndarray,
    x_from: np.ndarray,
) -> np.ndarray:
    """
    Return an (m, n) matrix mapping values defined on n points `x_from` to values
    defined on m points `x_to`, using Lagrange interpolation.

    For some elementwise scalar function ``f``, ``f(x_to) \\approx LM f(x_from)``.
    """
    assert len(x_to.shape) == 1, "x_to must be 1D."
    assert len(x_from.shape) == 1, "x_from must be 1D."
    num = np.broadcast_to(
        x_to[:, np.newaxis, np.newaxis] - x_from[np.newaxis, np.newaxis, :],
        (x_to.shape[0], x_from.shape[0], x_from.shape[0]),
    ).copy()
    num[:, np.arange(x_from.shape[0]), np.arange(x_from.shape[0])] = 1.0
    den = x_from[:, np.newaxis] - x_from[np.newaxis, :]
    np.fill_diagonal(den, 1.0)
    return num.prod(axis=-1)/den.prod(axis=-1)[np.newaxis, :]


def lagrangeprime_matrix(
    x_to: np.ndarray,
    x_from: np.ndarray,
) -> np.ndarray:
    """
    Return an (m, n) matrix of first derivatives of lagrange polynomials defined
    on n points `x_from` evaluated on m points `x_to`.
    """
    assert len(x_to.shape) == 1, "x_to must be 1D."
    assert len(x_from.shape) == 1, "x_from must be 1D."
    num = np.broadcast_to(
        x_to[:, np.newaxis, np.newaxis, np.newaxis] - x_from[np.newaxis, np.newaxis, np.newaxis, :],
        (x_to.shape[0], x_from.shape[0], x_from.shape[0], x_from.shape[0]),
    ).copy()
    a = np.arange(x_from.shape[0])
    num[:, a, :, a] = 1.0
    num[:, :, a, a] = 1.0
    den = x_from[:, np.newaxis] - x_from[np.newaxis, :]
    np.fill_diagonal(den, 1.0)
    p = num.prod(axis=-1)/den.prod(axis=-1)[np.newaxis, :, np.newaxis]
    p[:, a, a] = 0.0
    return p.sum(axis=-1)


def intfromsol_matrix(
    x_sp: np.ndarray,
    x_fp: np.ndarray,
) -> np.ndarray:
    """
    Return an (m, m) matrix to transform values defined on m points `x_sp` to mean values defined
    on m control volumes bounded by m + 1 points `x_fp`.
    """
    assert len(x_sp.shape) == 1, "x_sp must be 1D."
    assert len(x_fp.shape) == 1, "x_fp must be 1D."
    m = x_fp.shape[0] - x_sp.shape[0]
    n = x_sp.shape[0] - 1
    if n == 0:
        return np.ones((1, 1))
    x, w = gauss_legendre_quadrature(0.0, 1.0, n)
    # XXX: it would be nicer to allow 2D arrays in lagrange_matrix... here ravel then reshape instead.
    xi = (x_fp[:-1, np.newaxis] + np.diff(x_fp)[:, np.newaxis]*x[np.newaxis, :]).ravel()
    lm = lagrange_matrix(xi, x_sp).reshape(n + m, n, n + 1)
    return np.einsum("ijk,j->ik", lm, w)


def ader_matrix(
    x_time: np.ndarray,
    w_time: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Return the ADER (n, n) matrix defined using the n `x_time` points and `w_time` weigths on the [0, dt] interval.
    """
    assert len(x_time.shape) == 1, "x_time must be 1D."
    assert len(w_time.shape) == 1, "w_time must be 1D."
    assert x_time.shape == w_time.shape, "x_time and w_time shapes must match."
    ltm = lagrange_matrix(np.array([dt]), x_time).ravel()
    lpm = lagrangeprime_matrix(x_time, x_time)
    return ltm[np.newaxis, :]*ltm[:, np.newaxis] - lpm.T*w_time[np.newaxis, :]


def quadrature_mean(
    mesh: np.ndarray,
    fct,
    v: int,
) -> np.ndarray:
    """
    Return an array containing mean values of `fct` inside mesh control volumes.
    Means are calculated with a Gauss-Legendre quadrature of degree p.
    """
    p = mesh.shape[-1] - 1
    na = np.newaxis
    x, w = gauss_legendre_quadrature(0.0, 1.0, p)
    pts = np.ndarray((2, mesh.shape[1], mesh.shape[2], p, p, p, p))
    
    pts[0,...] = mesh[0,:,:,:-1,:-1,na,na] + x[na,na,na,na,na,:]*np.diff(mesh[0],axis=-1)[:,:,:-1,:,na,na]
    pts[1,...] = mesh[1,:,:,:-1,:-1,na,na] + x[na,na,na,na,:,na]*np.diff(mesh[1],axis=-2)[:,:,:,:-1,na,na]
    
    return np.einsum("x,y,ijklxy->ijkl",w,w,fct(pts,v))

