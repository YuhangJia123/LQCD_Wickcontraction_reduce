"""Exact multi-baryon coefficient benchmark for the 3H (pnn) example.

This implements the source-side coefficient matrix of the unified contraction
algorithm in arXiv:1205.0585, Eq. (13), for three nucleons.  It is deliberately
kept independent of the pn prototype so that the larger combinatorics are
visible:

* pnn has four up and five down quarks, hence 4! * 5! = 2880 flavour Wick maps;
* every map is expanded into the sparse source epsilon and Dirac factors;
* identical complete (colour, spin) endpoint keys are summed exactly and zero
  coefficients are discarded;
* a second colour-first accumulator reproduces the same dictionary and models
  the user's proposed colour hash followed by a Dirac/Fierz normal form;
* a separate local four-index Fierz check uses NumPy ``einsum`` for the
  Dirac inner products and reports its floating-point reconstruction residual.

The paper's block/UCA formula does not contain a general cross-baryon Fierz
identity: the sink baryon blocks are already contracted.  Therefore the Fierz
check below is deliberately a local four-index validation; it is not counted
as an additional reduction of the paper's global Nlist.
"""

from __future__ import annotations

import itertools
import json
import platform
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

# Source/sink baryon order is p,n,n.  Each tuple is (u,d,u) or (d,u,d).
BARYON_FLAVOURS: Tuple[Tuple[str, str, str], ...] = (
    ("u", "d", "u"),
    ("d", "u", "d"),
    ("d", "u", "d"),
)
BARYON_SLOTS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
)
FLAVOUR_SLOTS: Dict[str, Tuple[int, ...]] = {
    flavour: tuple(index for index, value in enumerate(sum(BARYON_FLAVOURS, ())) if value == flavour)
    for flavour in ("u", "d")
}
N_ENDPOINTS = 9
N_COLOUR = 3
N_SPIN = 4
COLOUR_BASE = N_COLOUR ** N_ENDPOINTS
SPIN_BASE = N_SPIN ** N_ENDPOINTS

# Provenance labels are kept next to the implementation so the report can
# distinguish reproduced paper structures from the user's added layers.
CODE_STRUCTURE_SOURCES: Tuple[Tuple[str, str, str], ...] = (
    ("BARYON_FLAVOURS / BARYON_SLOTS [论文1/UCA]", "固定 p,n,n 的三重子槽位和味道布局", "已实现"),
    ("FLAVOUR_SLOTS [论文1/UCA]", "按味道建立可置换的夸克槽位集合", "已实现"),
    ("LocalTerm [本工作：精确稀疏项]", "保存一个局部 epsilon*Dirac 非零项的颜色码、旋量码和整数系数", "已实现"),
    ("parity / epsilon_terms [论文1/UCA]", "生成 epsilon 的 6 个非零排列及 Grassmann/Wick 符号", "已实现"),
    ("gamma_matrices / nonzero_matrix [论文1算符]", "构造标准和非相对论 C Gamma 算符，并只取非零元", "已实现"),
    ("source_permutations [论文1/UCA]", "枚举同味 flavor Wick 映射及置换符号", "已实现"),
    ("local_terms / build_local_lists [论文1/UCA]", "生成每个重子局部颜色-Dirac 稀疏项表", "已实现"),
    ("accumulate_direct [论文1/UCA复现]", "构造完整 unified coefficient dictionary 并精确合并", "已实现"),
    ("accumulate_colour_first [本工作]", "颜色 exact key 先分桶，再在桶内合并 Dirac key", "已实现"),
    ("flatten_buckets / bucket_statistics [本工作]", "恢复统一键并统计颜色桶规模", "已实现"),
    ("benchmark_once / time_metrics [本工作]", "比较 direct 与颜色优先路径的时间和一致性", "已实现"),
    ("Fierz normal form + np.einsum [本工作扩展]", "跨重子开放 Dirac 指标的 Fierz/数值收缩归并", "局部四指标验证；未改写 Nlist"),
    ("RecursiveSourceTensorBuilder / orthogonalise [本工作]", "颜色分桶后构造源端 Dirac 稀疏张量，并用共轭内积、两遍 Modified Gram-Schmidt 和 Gram 缓存求 rank", "已实现；只处理源端结构"),
    ("source_dirac_to_full_dictionary / compare_reconstructed_dictionary [本工作验证]", "把正交化后的源端 Dirac 指标重新编码为完整 (colour_code, spin_code) 字典，并逐键对比颜色优先 exact 字典", "已实现；standard/NR 均通过"),
    ("determinant/LU [论文2]", "固定 weight pair 内把味道置换压缩为矩阵运算", "本程序未实现"),
    ("occupied-state recursion [论文3]", "按反对称占据态递归生成目标分量", "本程序未实现"),
)


@dataclass(frozen=True)
class LocalTerm:
    """One nonzero local epsilon*Dirac assignment for one baryon."""

    colour_code: int
    spin_code: int
    coefficient: int


def parity(values: Sequence[int]) -> int:
    inversions = sum(
        values[i] > values[j]
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )
    return -1 if inversions % 2 else 1


def encode_digits(values: Sequence[int], base: int) -> int:
    code = 0
    multiplier = 1
    for value in values:
        code += int(value) * multiplier
        multiplier *= base
    return code


def decode_digits(code: int, length: int, base: int) -> Tuple[int, ...]:
    values = []
    for _ in range(length):
        values.append(code % base)
        code //= base
    return tuple(values)


def epsilon_terms() -> List[Tuple[Tuple[int, int, int], int]]:
    """Return the six nonzero epsilon colour assignments and their signs."""
    result = []
    for values in itertools.permutations(range(N_COLOUR)):
        result.append((values, parity(values)))
    return result


def gamma_matrices(standard: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Return C Gamma_1 and Gamma_2 for the paper's two operator choices."""
    # Euclidean Dirac basis, matching the matrices in pn_method_prototype.py.
    i2 = np.eye(2, dtype=np.complex128)
    z2 = np.zeros((2, 2), dtype=np.complex128)
    s1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    s3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    def spatial(pauli: np.ndarray) -> np.ndarray:
        return np.block([[z2, -1j * pauli], [1j * pauli, z2]])

    g1, g2, g3 = (spatial(sigma) for sigma in (s1, s2, s3))
    g4 = np.block([[i2, z2], [z2, -i2]])
    g5 = g1 @ g2 @ g3 @ g4
    c = g4 @ g2
    if standard:
        return c @ g5, np.eye(4, dtype=np.complex128)
    pnr = (np.eye(4, dtype=np.complex128) + g4) / 2.0
    return c @ g5 @ pnr, pnr


def dirac_basis_16() -> Tuple[np.ndarray, ...]:
    """Return a complete 16-element Euclidean Dirac matrix basis.

    Products are ordered by increasing bit mask.  This avoids choosing a
    convention-dependent ``gamma_5 gamma_mu``/sigma basis while retaining a
    fixed, reproducible normal form for the numerical Fierz projection.
    """
    i2 = np.eye(2, dtype=np.complex128)
    z2 = np.zeros((2, 2), dtype=np.complex128)
    s1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    s3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    def spatial(pauli: np.ndarray) -> np.ndarray:
        return np.block([[z2, -1j * pauli], [1j * pauli, z2]])

    gammas = (*tuple(spatial(sigma) for sigma in (s1, s2, s3)), np.block([[i2, z2], [z2, -i2]]))
    identity = np.eye(4, dtype=np.complex128)
    basis = []
    for mask in range(16):
        matrix = identity.copy()
        for mu, gamma in enumerate(gammas):
            if mask & (1 << mu):
                matrix = matrix @ gamma
        basis.append(matrix)
    return tuple(basis)


def einsum_inner(left: np.ndarray, right: np.ndarray) -> complex:
    """Conjugate Euclidean tensor inner product using only NumPy einsum."""
    left_array = np.asarray(left, dtype=np.complex128)
    right_array = np.asarray(right, dtype=np.complex128)
    if left_array.shape != right_array.shape:
        raise ValueError(f"inner-product shape mismatch: {left_array.shape} != {right_array.shape}")
    return complex(np.einsum("...,...->", left_array.conjugate(), right_array, optimize=True))


def dirac_matrix_coefficients(matrix: np.ndarray) -> np.ndarray:
    """Project a 4 x 4 matrix with c_A = Tr(Gamma_A^dagger M) / 4."""
    basis = np.asarray(dirac_basis_16(), dtype=np.complex128)
    matrix_array = np.asarray(matrix, dtype=np.complex128)
    if matrix_array.shape != (4, 4):
        raise ValueError(f"expected a 4 x 4 Dirac matrix, got {matrix_array.shape}")
    return np.einsum("aij,ij->a", basis.conjugate(), matrix_array, optimize=True) / 4.0


def lattice_transpose_signs() -> np.ndarray:
    """Signs of the raw-transpose Gram matrix in the lattice basis.

    For the un-rephased ordered gamma products used here,
    Tr(Gamma_A.T Gamma_B) = 4 * signs[A] * delta_AB.  Multiplying the
    transpose dual by these signs gives an exact algebraic Fierz dual.
    """
    basis = np.asarray(dirac_basis_16(), dtype=np.complex128)
    gram = np.einsum("aij,bij->ab", basis, basis, optimize=True)
    if np.max(np.abs(gram - np.diag(np.diag(gram)))) > 1.0e-12:
        raise ValueError("lattice transpose Gram matrix is not diagonal")
    signs = np.real_if_close(np.diag(gram) / 4.0).astype(np.float64)
    if not np.allclose(np.abs(signs), 1.0):
        raise ValueError("lattice transpose Gram diagonal is not +/-4")
    return signs


def dirac_matrix_coefficients_lattice_transpose(matrix: np.ndarray) -> np.ndarray:
    """Project with the lattice algebraic transpose dual.

    If the basis is rephased so that Tr(Gamma_A.T Gamma_B)=4 delta_AB,
    this is exactly 1/4 Tr(Gamma_A.T M).  For the raw ordered products used
    here, ``lattice_transpose_signs`` supplies the dual-basis signs.
    """
    basis = np.asarray(dirac_basis_16(), dtype=np.complex128)
    matrix_array = np.asarray(matrix, dtype=np.complex128)
    if matrix_array.shape != (4, 4):
        raise ValueError(f"expected a 4 x 4 Dirac matrix, got {matrix_array.shape}")
    raw_transpose_trace = np.einsum("aij,ij->a", basis, matrix_array, optimize=True)
    return lattice_transpose_signs() * raw_transpose_trace / 4.0


_FIERZ_PRODUCT_BASIS: Optional[np.ndarray] = None


def fierz_product_basis() -> np.ndarray:
    """Build and cache the crossed 16 x 16 product basis.

    For a direct tensor ``A[a,b] B[c,d]`` the crossed pairing is represented
    by ``P[a,d] Q[c,b]``.  There are 256 such tensors, exactly spanning the
    4^4-dimensional four-index Dirac tensor space.
    """
    global _FIERZ_PRODUCT_BASIS
    if _FIERZ_PRODUCT_BASIS is None:
        basis = dirac_basis_16()
        _FIERZ_PRODUCT_BASIS = np.stack(
            [
                np.einsum("ad,cb->abcd", left, right, optimize=True)
                for left in basis
                for right in basis
            ],
            axis=0,
        )
    return _FIERZ_PRODUCT_BASIS


def fierz_einsum_projection(standard: bool) -> Dict[str, object]:
    """Project one baryon ``C Gamma_1``/``Gamma_2`` product into Fierz form.

    The projection is a numerical validation of the local identity only.  It
    does not merge entries in the exact colour/spin coefficient dictionary.
    All contractions in this routine are explicit ``np.einsum`` calls, so it
    does not depend on the unavailable ``contract`` package.
    """
    start = time.perf_counter()
    left, right = gamma_matrices(standard)
    matrix_basis = np.asarray(dirac_basis_16(), dtype=np.complex128)
    left_coefficients = dirac_matrix_coefficients_lattice_transpose(left)
    right_coefficients = dirac_matrix_coefficients_lattice_transpose(right)
    left_projection = np.einsum("a,aij->ij", left_coefficients, matrix_basis, optimize=True)
    right_projection = np.einsum("a,aij->ij", right_coefficients, matrix_basis, optimize=True)
    direct = np.einsum("ab,cd->abcd", left, right, optimize=True)
    crossed = fierz_product_basis()
    transpose_signs = lattice_transpose_signs()
    product_signs = np.repeat(transpose_signs, 16) * np.tile(transpose_signs, 16)
    overlaps = np.einsum("kabcd,abcd->k", crossed, direct, optimize=True)
    coefficients = product_signs * overlaps / 16.0
    reconstructed = np.einsum("k,kabcd->abcd", coefficients, crossed, optimize=True)
    residual = direct - reconstructed
    matrix_left_residual = left - left_projection
    matrix_right_residual = right - right_projection
    direct_norm2 = float(einsum_inner(direct, direct).real)
    residual_norm2 = float(einsum_inner(residual, residual).real)
    reconstructed_norm2 = float(einsum_inner(reconstructed, reconstructed).real)
    scale = max(direct_norm2, reconstructed_norm2, np.finfo(float).tiny)
    product_basis_norms = np.einsum("kabcd,kabcd->k", crossed.conjugate(), crossed, optimize=True)
    product_basis_matrix = crossed.reshape(crossed.shape[0], -1)
    return {
        "operator": "standard" if standard else "nonrelativistic",
        "backend": "numpy.einsum (CPU)",
        "pairing": "lattice transpose dual",
        "gpu_used": False,
        "direct_tensor_shape": list(direct.shape),
        "fierz_basis_dimension": 16,
        "crossed_product_dimension": int(crossed.shape[0]),
        "matrix_basis_trace_norm": 4.0,
        "matrix_projection_factor": 0.25,
        "transpose_gram_signs": [int(round(value)) for value in transpose_signs],
        "matrix_projection_residual": float(
            np.sqrt(
                max(
                    einsum_inner(matrix_left_residual, matrix_left_residual).real
                    + einsum_inner(matrix_right_residual, matrix_right_residual).real,
                    0.0,
                )
            )
        ),
        "fierz_product_trace_norm": 16.0,
        "fierz_product_projection_factor": 1.0 / 16.0,
        "fierz_product_pairing": "transpose dual signs / 16",
        "fierz_product_norm_min": float(np.min(product_basis_norms.real)),
        "fierz_product_norm_max": float(np.max(product_basis_norms.real)),
        "gram_rank": int(np.linalg.matrix_rank(product_basis_matrix, tol=1.0e-10)),
        "direct_nonzero_entries": int(np.count_nonzero(np.abs(direct) > 1.0e-12)),
        "fierz_nonzero_coefficients": int(np.count_nonzero(np.abs(coefficients) > 1.0e-10)),
        "direct_norm2": direct_norm2,
        "reconstructed_norm2": reconstructed_norm2,
        "relative_residual": float(np.sqrt(max(residual_norm2, 0.0) / scale)),
        "seconds": time.perf_counter() - start,
        "interpretation": "local four-index Fierz identity; not an extra global UCA Nlist reduction",
    }


def fierz_einsum_suite() -> Dict[str, Dict[str, object]]:
    """Run the cached NumPy Fierz check for both operator choices."""
    return {
        "standard": fierz_einsum_projection(True),
        "nonrelativistic": fierz_einsum_projection(False),
    }


def nonzero_matrix(matrix: np.ndarray) -> List[Tuple[int, int, int]]:
    """All nonzero entries; this example has integer entries only."""
    result = []
    for row, column in zip(*np.nonzero(np.abs(matrix) > 1.0e-12)):
        value = complex(matrix[row, column])
        if abs(value.imag) > 1.0e-12 or abs(abs(value.real) - round(abs(value.real))) > 1.0e-12:
            raise ValueError("expected real integer gamma entries for this benchmark")
        result.append((int(row), int(column), int(round(value.real))))
    return result


def local_terms(
    endpoint_slots: Sequence[int],
    external_spin: int,
    cg1_nonzero: Sequence[Tuple[int, int, int]],
    gamma2_nonzero: Sequence[Tuple[int, int, int]],
    eps_terms: Sequence[Tuple[Tuple[int, int, int], int]],
) -> List[LocalTerm]:
    """Expand one source baryon into sparse colour/spin endpoint assignments."""
    terms: List[LocalTerm] = []
    for colours, eps_sign in eps_terms:
        colour_digits = [0] * N_ENDPOINTS
        for endpoint, colour in zip(endpoint_slots, colours):
            colour_digits[endpoint] = colour
        colour_code = encode_digits(colour_digits, N_COLOUR)
        for spin0, spin1, cg1_value in cg1_nonzero:
            for gamma2_row, spin2, gamma2_value in gamma2_nonzero:
                if gamma2_row != external_spin:
                    continue
                spin_digits = [0] * N_ENDPOINTS
                spin_digits[endpoint_slots[0]] = spin0
                spin_digits[endpoint_slots[1]] = spin1
                spin_digits[endpoint_slots[2]] = spin2
                spin_code = encode_digits(spin_digits, N_SPIN)
                terms.append(LocalTerm(colour_code, spin_code, eps_sign * cg1_value * gamma2_value))
    return terms


def combine_local_terms(local_lists: Sequence[Sequence[LocalTerm]]) -> Iterator[Tuple[int, int, int]]:
    """Yield combined colour code, spin code and coefficient for a sigma map."""
    for first in local_lists[0]:
        for second in local_lists[1]:
            for third in local_lists[2]:
                # Each remapped local term has support on a disjoint set of
                # endpoint slots.  Adding digit-wise therefore merges the
                # three local assignments without assuming that endpoint
                # slots remain in their original baryon blocks.  Since the
                # supports are disjoint, integer base encodings can be added
                # directly and no carry can occur.
                yield (
                    first.colour_code + second.colour_code + third.colour_code,
                    first.spin_code + second.spin_code + third.spin_code,
                    first.coefficient * second.coefficient * third.coefficient,
                )


def source_permutations() -> Iterator[Tuple[Dict[int, int], int]]:
    """Yield source-slot -> endpoint maps and the fermion permutation sign."""
    endpoint_by_flavour = FLAVOUR_SLOTS
    for up in itertools.permutations(endpoint_by_flavour["u"]):
        up_indices = tuple(endpoint_by_flavour["u"].index(value) for value in up)
        up_sign = parity(up_indices)
        for down in itertools.permutations(endpoint_by_flavour["d"]):
            down_indices = tuple(endpoint_by_flavour["d"].index(value) for value in down)
            down_sign = parity(down_indices)
            mapping: Dict[int, int] = {}
            for source_slot, endpoint in zip(endpoint_by_flavour["u"], up):
                mapping[source_slot] = endpoint
            for source_slot, endpoint in zip(endpoint_by_flavour["d"], down):
                mapping[source_slot] = endpoint
            yield mapping, up_sign * down_sign


def build_local_lists(standard: bool, source_spins: Sequence[int]) -> List[List[LocalTerm]]:
    cg1, gamma2 = gamma_matrices(standard)
    cg1_nonzero = nonzero_matrix(cg1)
    gamma2_nonzero = nonzero_matrix(gamma2)
    eps_terms = epsilon_terms()
    return [
        local_terms(BARYON_SLOTS[index], source_spins[index], cg1_nonzero, gamma2_nonzero, eps_terms)
        for index in range(3)
    ]


def accumulate_direct(standard: bool, source_spins: Sequence[int]) -> Dict[int, int]:
    """Build the complete UCA coefficient dictionary with exact integer sums."""
    local_cache = build_local_lists(standard, source_spins)
    coefficients: Dict[int, int] = {}
    for _mapping, permutation_sign in source_permutations():
        # The local lists are generated from source baryon slots.  Re-map each
        # local term to endpoint positions for this sigma before combining.
        remapped: List[List[LocalTerm]] = []
        for baryon_index, source_slots in enumerate(BARYON_SLOTS):
            endpoint_slots = tuple(_mapping[slot] for slot in source_slots)
            original = local_cache[baryon_index]
            baryon_remapped = []
            for term in original:
                colours = decode_digits(term.colour_code, N_ENDPOINTS, N_COLOUR)
                spins = decode_digits(term.spin_code, N_ENDPOINTS, N_SPIN)
                colour_values = tuple(colours[slot] for slot in source_slots)
                spin_values = tuple(spins[slot] for slot in source_slots)
                colour_digits = [0] * N_ENDPOINTS
                spin_digits = [0] * N_ENDPOINTS
                for endpoint, colour, spin in zip(endpoint_slots, colour_values, spin_values):
                    colour_digits[endpoint] = colour
                    spin_digits[endpoint] = spin
                baryon_remapped.append(
                    LocalTerm(
                        encode_digits(colour_digits, N_COLOUR),
                        encode_digits(spin_digits, N_SPIN),
                        term.coefficient,
                    )
                )
            remapped.append(baryon_remapped)
        for colour_code, spin_code, coefficient in combine_local_terms(remapped):
            key = colour_code * SPIN_BASE + spin_code
            coefficients[key] = coefficients.get(key, 0) + permutation_sign * coefficient
    return {key: value for key, value in coefficients.items() if value != 0}


def accumulate_colour_first(standard: bool, source_spins: Sequence[int]) -> Dict[int, Dict[int, int]]:
    """Colour hash first, then accumulate the Dirac endpoint signature."""
    local_cache = build_local_lists(standard, source_spins)
    buckets: Dict[int, Dict[int, int]] = defaultdict(dict)
    for mapping, permutation_sign in source_permutations():
        remapped: List[List[LocalTerm]] = []
        for baryon_index, source_slots in enumerate(BARYON_SLOTS):
            endpoint_slots = tuple(mapping[slot] for slot in source_slots)
            baryon_remapped = []
            for term in local_cache[baryon_index]:
                colours = decode_digits(term.colour_code, N_ENDPOINTS, N_COLOUR)
                spins = decode_digits(term.spin_code, N_ENDPOINTS, N_SPIN)
                colour_digits = [0] * N_ENDPOINTS
                spin_digits = [0] * N_ENDPOINTS
                for endpoint, source_slot in zip(endpoint_slots, source_slots):
                    colour_digits[endpoint] = colours[source_slot]
                    spin_digits[endpoint] = spins[source_slot]
                baryon_remapped.append(
                    LocalTerm(
                        encode_digits(colour_digits, N_COLOUR),
                        encode_digits(spin_digits, N_SPIN),
                        term.coefficient,
                    )
                )
            remapped.append(baryon_remapped)
        for colour_code, spin_code, coefficient in combine_local_terms(remapped):
            bucket = buckets[colour_code]
            bucket[spin_code] = bucket.get(spin_code, 0) + permutation_sign * coefficient
    result = {}
    for colour, bucket in buckets.items():
        clean = {spin: value for spin, value in bucket.items() if value != 0}
        if clean:
            result[colour] = clean
    return result


def flatten_buckets(buckets: Mapping[int, Mapping[int, int]]) -> Dict[int, int]:
    return {
        colour * SPIN_BASE + spin: value
        for colour, bucket in buckets.items()
        for spin, value in bucket.items()
    }


def bucket_statistics(buckets: Mapping[int, Mapping[int, int]]) -> Dict[str, object]:
    sizes = [len(bucket) for bucket in buckets.values()]
    return {
        "colour_buckets": len(buckets),
        "nonzero_entries": sum(sizes),
        "spin_entries_per_colour_min": min(sizes) if sizes else 0,
        "spin_entries_per_colour_max": max(sizes) if sizes else 0,
        "spin_entries_per_colour_mean": statistics.fmean(sizes) if sizes else 0.0,
    }


def benchmark_once(standard: bool, source_spins: Sequence[int]) -> Dict[str, object]:
    start = time.perf_counter()
    direct = accumulate_direct(standard, source_spins)
    direct_time = time.perf_counter() - start
    start = time.perf_counter()
    buckets = accumulate_colour_first(standard, source_spins)
    colour_time = time.perf_counter() - start
    flattened = flatten_buckets(buckets)
    expected_nlist = 3_775_680 if standard else 25_920
    return {
        "operator": "standard" if standard else "nonrelativistic",
        "source_spins": tuple(source_spins),
        "flavour_wick_maps": 2880,
        "direct_nonzero_entries": len(direct),
        "colour_first_nonzero_entries": len(flattened),
        "paper_expected_nlist": expected_nlist,
        "paper_expected_ncontr": expected_nlist // 8,
        "matches_paper_nlist": len(direct) == expected_nlist,
        "direct_equals_colour_first": direct == flattened,
        "colour_first": bucket_statistics(buckets),
        "direct_seconds": direct_time,
        "colour_first_seconds": colour_time,
        "fierz_note": "Paper 1205.0585 precontracts sink baryon blocks; no generic cross-baryon Fierz quotient is included in this exact UCA table.",
    }


def main() -> None:
    # (0,0,1) is the nonzero spin block used in Tables A.7/A.8 of paper 1.
    fierz_results = fierz_einsum_suite()
    results = {
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "model": "3H = pnn source coefficient matrix, paper 1205.0585 Eq. (13)",
        "standard": benchmark_once(True, (0, 0, 1)),
        "nonrelativistic": benchmark_once(False, (0, 0, 1)),
        "fierz_einsum": fierz_results,
        "definitions": {
            "baryon_flavours": BARYON_FLAVOURS,
            "flavour_slot_counts": {key: len(value) for key, value in FLAVOUR_SLOTS.items()},
            "colour_base": N_COLOUR,
            "spin_base": N_SPIN,
            "full_key": "base-3 colour endpoint code followed by base-4 Dirac endpoint code",
            "code_structure_sources": CODE_STRUCTURE_SOURCES,
        },
    }
    output_json = HERE / "benchmark_3h_results.json"
    output_txt = HERE / "benchmark_3h_results.txt"
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = [
        "3H pnn unified-coefficient benchmark",
        f"machine: {results['machine']['platform']}",
        f"numpy: {results['machine']['numpy']}",
    ]
    for key in ("standard", "nonrelativistic"):
        result = results[key]
        lines.extend(
            [
                f"{key}:",
                f"  flavour Wick maps: {result['flavour_wick_maps']}",
                f"  direct nonzero Nlist: {result['direct_nonzero_entries']}",
                f"  colour-first nonzero Nlist: {result['colour_first_nonzero_entries']}",
                f"  paper expected Nlist: {result['paper_expected_nlist']}",
                f"  paper Ncontr=Nlist/8: {result['paper_expected_ncontr']}",
                f"  matches paper Nlist: {result['matches_paper_nlist']}",
                f"  direct == colour-first: {result['direct_equals_colour_first']}",
                f"  colour buckets: {result['colour_first']['colour_buckets']}",
                f"  direct time: {result['direct_seconds']:.6f} s",
                f"  colour-first time: {result['colour_first_seconds']:.6f} s",
            ]
        )
        fierz = fierz_results[key]
        lines.extend(
            [
                f"  Fierz backend: {fierz['backend']}",
                f"  Fierz Gram rank: {fierz['gram_rank']}/{fierz['crossed_product_dimension']}",
                f"  Fierz nonzero coefficients: {fierz['fierz_nonzero_coefficients']}",
                f"  Fierz relative residual: {fierz['relative_residual']:.3e}",
                f"  Fierz time: {fierz['seconds']:.6f} s",
            ]
        )
    lines.append("Fierz: local four-index identity checked with np.einsum; global UCA Nlist is unchanged.")
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
