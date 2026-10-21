from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class KernelParams:
    mu: np.ndarray
    lam: np.ndarray
    a_1: np.ndarray
    a_2: np.ndarray
    b_1: np.ndarray
    b_2: np.ndarray
    f: np.ndarray
    q: float


@dataclass
class LinearizedFunctions:
    vs: float
    linearization_term: float
    tau: float
    q: float
    road_length_m: float = 500.0

    def lam(self, x):
        return np.zeros_like(np.asarray(x, dtype=float)) + self.vs

    def mu(self, x):
        return np.zeros_like(np.asarray(x, dtype=float)) + self.linearization_term - self.vs

    def lambda_d(self, x):
        return np.zeros_like(np.asarray(x, dtype=float))

    def mu_d(self, x):
        return np.zeros_like(np.asarray(x, dtype=float))

    def c_1(self, x):
        return np.zeros_like(np.asarray(x, dtype=float))

    def c_2(self, x):
        x = np.asarray(x, dtype=float)
        return -1.0 / self.tau * np.exp(-x / (self.tau * self.vs))


def solve_kernel_2x2(params: KernelParams, n_g: int) -> tuple[np.ndarray, np.ndarray]:
    """Historical generic unit-domain solver; not used by the revision pipeline."""
    delta = 1.0 / (n_g - 1)
    num_tot = n_g * (n_g + 1) // 2

    index_map = np.zeros((n_g, n_g), dtype=int)
    counter = 1
    for col in range(n_g):
        for row in range(col + 1):
            index_map[row, col] = counter
            counter += 1

    glob_to_mat = []
    for col in range(n_g):
        for row in range(col + 1):
            glob_to_mat.append((row, col))

    indexes = np.tile(np.arange(1, 2 * num_tot + 1, dtype=int)[:, None], (1, 5))
    weights = np.zeros((2 * num_tot, 5), dtype=float)
    rhs = np.zeros(2 * num_tot, dtype=float)

    s_k_x, s_k_y = np.meshgrid(params.mu, -params.lam)
    s_l_x, s_l_y = np.meshgrid(params.mu, params.mu)
    s_k = np.sqrt(s_k_x**2 + s_k_y**2)
    s_l = np.sqrt(s_l_x**2 + s_l_y**2)

    for ind_x in range(n_g):
        for ind_y in range(ind_x + 1):
            glob_ind = index_map[ind_y, ind_x] - 1
            if ind_y == ind_x:
                weights[glob_ind, 0] = 1.0
                rhs[glob_ind] = params.f[ind_y]
            else:
                theta = np.arctan2(s_k_x[ind_y, ind_x], -s_k_y[ind_y, ind_x])
                if theta > np.pi / 4.0:
                    sigma = delta / np.sin(theta)
                    d = sigma * np.cos(theta)
                    prefactor = s_k[ind_y, ind_x] / sigma
                    weights[glob_ind, 0] = prefactor
                    weights[glob_ind, 1] = -prefactor * d / delta
                    indexes[glob_ind, 1] = index_map[ind_y + 1, ind_x - 1]
                    weights[glob_ind, 2] = -prefactor * (delta - d) / delta
                    indexes[glob_ind, 2] = index_map[ind_y, ind_x - 1]
                else:
                    sigma = delta / np.cos(theta)
                    d = sigma * np.sin(theta)
                    prefactor = s_k[ind_y, ind_x] / sigma
                    weights[glob_ind, 0] = prefactor
                    weights[glob_ind, 1] = -prefactor * d / delta
                    indexes[glob_ind, 1] = index_map[ind_y + 1, ind_x - 1]
                    weights[glob_ind, 2] = -prefactor * (delta - d) / delta
                    indexes[glob_ind, 2] = index_map[ind_y + 1, ind_x]

                if ind_y == ind_x - 1:
                    weights[glob_ind, 2] += weights[glob_ind, 1]
                    weights[glob_ind, 0] -= weights[glob_ind, 1]
                    if theta > np.pi / 4.0:
                        indexes[glob_ind, 1] = index_map[ind_y + 1, ind_x]
                    else:
                        indexes[glob_ind, 1] = index_map[ind_y, ind_x - 1]

    for ind_x in range(n_g):
        for ind_y in range(ind_x + 1):
            glob_ind = index_map[ind_y, ind_x] - 1 + num_tot
            if ind_y == 0:
                weights[glob_ind, 0] = -1.0
                weights[glob_ind, 1] = params.q
                indexes[glob_ind, 1] = glob_ind + 1 - num_tot
            else:
                theta = np.arctan2(s_l_y[ind_y, ind_x], s_l_x[ind_y, ind_x])
                if theta < np.pi / 4.0:
                    sigma = delta / np.cos(theta)
                    d = sigma * np.sin(theta)
                    prefactor = s_l[ind_y, ind_x] / sigma
                    weights[glob_ind, 0] = prefactor
                    weights[glob_ind, 1] = -prefactor * (delta - d) / delta
                    indexes[glob_ind, 1] = index_map[ind_y, ind_x - 1] + num_tot
                    weights[glob_ind, 2] = -prefactor * d / delta
                    indexes[glob_ind, 2] = index_map[ind_y - 1, ind_x - 1] + num_tot
                else:
                    sigma = delta / np.sin(theta)
                    d = sigma * np.cos(theta)
                    prefactor = s_l[ind_y, ind_x] / sigma
                    weights[glob_ind, 0] = prefactor
                    weights[glob_ind, 1] = -prefactor * d / delta
                    indexes[glob_ind, 1] = index_map[ind_y - 1, ind_x - 1] + num_tot
                    weights[glob_ind, 2] = -prefactor * (delta - d) / delta
                    indexes[glob_ind, 2] = index_map[ind_y - 1, ind_x] + num_tot

    for ind_x in range(n_g):
        for ind_y in range(ind_x + 1):
            glob_ind = index_map[ind_y, ind_x] - 1
            if ind_y != ind_x:
                weights[glob_ind, 3] = -params.a_1[ind_y, ind_x]
                weights[glob_ind, 4] = -params.a_2[ind_y, ind_x]
                indexes[glob_ind, 4] = glob_ind + 1 + num_tot
            if ind_y != 0:
                weights[glob_ind + num_tot, 3] = -params.b_1[ind_y, ind_x]
                indexes[glob_ind + num_tot, 3] = glob_ind + 1
                weights[glob_ind + num_tot, 4] = -params.b_2[ind_y, ind_x]

    row_idx = np.repeat(np.arange(2 * num_tot), 5)
    col_idx = indexes.reshape(-1) - 1
    vals = weights.reshape(-1)
    valid = col_idx >= 0
    matrix = sparse.coo_matrix((vals[valid], (row_idx[valid], col_idx[valid])), shape=(2 * num_tot, 2 * num_tot)).tocsr()
    sol = spsolve(matrix, rhs)

    k_mat = np.zeros((n_g, n_g), dtype=float)
    l_mat = np.zeros((n_g, n_g), dtype=float)
    k_dummy = sol[:num_tot]
    l_dummy = sol[num_tot:]
    for idx, (row, col) in enumerate(glob_to_mat):
        k_mat[row, col] = k_dummy[idx]
        l_mat[row, col] = l_dummy[idx]
    return k_mat, l_mat


def _mesh_params(fun: LinearizedFunctions, n_g: int, observer: bool = False) -> KernelParams:
    xspan = np.linspace(0.0, 1.0, n_g)
    lam = fun.lam(xspan)
    mu = fun.mu(xspan)
    a_1 = np.zeros((n_g, n_g))
    a_2 = np.zeros((n_g, n_g))
    b_1 = np.zeros((n_g, n_g))
    b_2 = np.zeros((n_g, n_g))
    for i in range(n_g):
        if observer:
            a_1[i, :] = -fun.lambda_d(xspan)
            a_2[i, :] = fun.c_2(xspan[i])
            b_1[i, :] = fun.c_1(xspan[i])
            b_2[i, :] = -fun.lambda_d(xspan)
        else:
            a_1[i, :] = fun.lambda_d(xspan[i])
            a_2[i, :] = fun.c_2(xspan[i])
            b_1[i, :] = fun.c_1(xspan[i])
            b_2[i, :] = -fun.mu_d(xspan[i])
    f = -fun.c_2(xspan) / (fun.lam(xspan) + fun.mu(xspan))
    q = fun.q if observer else fun.q * fun.lam(0.0) / fun.mu(0.0)
    return KernelParams(mu=mu, lam=lam, a_1=a_1, a_2=a_2, b_1=b_1, b_2=b_2, f=f, q=float(q))


def controller_gains(fun: LinearizedFunctions, n_g: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact physical-domain kernels for the constant-coefficient ARZ model.

    Arrays use rows xi, columns x, on 0 <= xi <= x <= L.
    K1=C exp(-xi/(tau*a)), K2=-C, C=1/(tau*(a+b)).
    Substitution satisfies both kernel PDEs and both boundary conditions.
    """
    grid = np.linspace(0, fun.road_length_m, n_g)
    c = 1.0 / (fun.tau * fun.linearization_term)
    triangle = np.triu(np.ones((n_g, n_g)))
    return c*np.exp(-grid[:, None]/(fun.tau*fun.vs))*triangle, -c*triangle


def observer_gains(fun: LinearizedFunctions, n_g: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact upper-domain M1,M2; rows x, columns xi, x <= xi <= L."""
    grid = np.linspace(0, fun.road_length_m, n_g)
    c = 1.0 / (fun.tau * fun.linearization_term)
    triangle = np.triu(np.ones((n_g, n_g)))
    return fun.q*c*triangle, c*np.exp(-grid[:, None]/(fun.tau*fun.vs))*triangle


def inverse_kernels(fun: LinearizedFunctions, x, xi):
    """Positive-sign inverse: v=beta+int L1 alpha+int L2 beta."""
    a, b = fun.vs, fun.linearization_term-fun.vs
    c = 1.0/(fun.tau*(a+b))
    x, xi = np.asarray(x), np.asarray(xi)
    return (c*np.exp(-(a*x+b*xi)/(fun.tau*a*(a+b))),
            -c*np.exp(-(x-xi)/(fun.tau*(a+b))))
