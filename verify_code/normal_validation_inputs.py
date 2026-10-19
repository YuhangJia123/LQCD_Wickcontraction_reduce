"""Shared normal-mode arrays for independent verification scripts."""

import itertools

import numpy as np

from kernels import correlator_wick_normalizer as core


def _epsilon_color():
    epsilon = np.zeros((3, 3, 3), dtype=np.complex128)
    for colors in itertools.permutations(range(3)):
        inversions = sum(
            colors[i] > colors[j]
            for i in range(3) for j in range(i + 1, 3)
        )
        epsilon[colors] = -1 if inversions % 2 else 1
    return epsilon


def _inputs(seed):
    rng = np.random.default_rng(seed)

    def peram():
        return (
            rng.standard_normal((3, 4, 3, 4))
            + 1j * rng.standard_normal((3, 4, 3, 4))
        ) / np.sqrt(2)

    gamma = core._euclidean_gamma_matrices()
    return {
        "C_g5": np.asarray(gamma["Cgamma5"], dtype=np.complex128),
        "C_g5_dagger": np.asarray(gamma["Cgamma5_dagger"], dtype=np.complex128),
        "peram_u_sin_sou_x_x": peram(),
        "peram_d_sin_sou_x_x": peram(),
        "epsilon_color": _epsilon_color(),
    }


__all__ = ["_epsilon_color", "_inputs"]
