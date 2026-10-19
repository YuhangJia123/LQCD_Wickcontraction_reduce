"""Small factorized meson/baryon fixtures used by the package examples."""

from __future__ import annotations

from typing import Sequence, Tuple

from .model import Correlator, Field, LocalVertex, Operator


Matrix = Sequence[Sequence[complex]]


IDENTITY = tuple(
    tuple(1 + 0j if row == column else 0j for column in range(4))
    for row in range(4)
)

CG5 = (
    (0j, 1 + 0j, 0j, 0j),
    (-1 + 0j, 0j, 0j, 0j),
    (0j, 0j, 0j, 1 + 0j),
    (0j, 0j, -1 + 0j, 0j),
)

GAMMA5 = (
    (0j, 0j, 1 + 0j, 0j),
    (0j, 0j, 0j, 1 + 0j),
    (1 + 0j, 0j, 0j, 0j),
    (0j, 1 + 0j, 0j, 0j),
)


def _matrix_terms(matrix: Matrix) -> Tuple[Tuple[Tuple[int, ...], complex], ...]:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("Dirac matrices must be 4 x 4")
    return tuple(
        ((row, column), complex(value))
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
        if value != 0
    )


def meson_vertex(
    name: str,
    *,
    antiquark_flavor: str,
    quark_flavor: str,
    gamma: Matrix,
) -> LocalVertex:
    """Make a colour-delta meson vertex with ordered ``[antiquark, quark]`` slots."""

    fields = (Field(antiquark_flavor, barred=True), Field(quark_flavor))
    return LocalVertex(name, fields, "delta", _matrix_terms(gamma))


def baryon_vertex(
    name: str,
    *,
    flavors: Sequence[str],
    barred: bool,
    external_spin: int,
    diquark_matrix: Matrix = CG5,
    open_matrix: Matrix = IDENTITY,
) -> LocalVertex:
    """Make an ordered epsilon vertex with a caller-selected source spin tensor."""

    if len(flavors) != 3:
        raise ValueError("a baryon vertex needs three ordered flavors")
    if not 0 <= external_spin < 4:
        raise ValueError("external_spin must be in [0, 3]")
    if any(len(matrix) != 4 or any(len(row) != 4 for row in matrix)
           for matrix in (diquark_matrix, open_matrix)):
        raise ValueError("Dirac matrices must be 4 x 4")

    terms = []
    for first in range(4):
        for second in range(4):
            first_value = complex(diquark_matrix[first][second])
            if not first_value:
                continue
            for third in range(4):
                third_value = complex(open_matrix[external_spin][third])
                if third_value:
                    terms.append(((first, second, third), first_value * third_value))

    fields = tuple(Field(flavor, barred=barred) for flavor in flavors)
    return LocalVertex(name, fields, "epsilon", tuple(terms))


def make_pnn() -> Correlator:
    p = ("u", "d", "u")
    n = ("d", "u", "d")
    sink_fields = tuple(Field(flavor) for triple in (p, n, n) for flavor in triple)
    source_vertices = (
        baryon_vertex("pbar", flavors=p, barred=True, external_spin=0),
        baryon_vertex("nbar_1", flavors=n, barred=True, external_spin=0),
        baryon_vertex("nbar_2", flavors=n, barred=True, external_spin=1),
    )
    return Correlator(
        Operator("pnn_sink", sink_fields),
        Operator.from_vertices("pnn_source", source_vertices),
    )


def make_pi_plus_pi_minus() -> Correlator:
    sink_fields = (Field("d", barred=True), Field("u"))
    source = meson_vertex(
        "pi_minus_source", antiquark_flavor="u", quark_flavor="d", gamma=GAMMA5
    )
    return Correlator(Operator("pi_plus_sink", sink_fields),
                      Operator.from_vertices("pi_minus_source", (source,)))


def make_p_pi() -> Correlator:
    p = ("u", "d", "u")
    sink_fields = tuple(Field(flavor) for flavor in p) + (
        Field("d", barred=True), Field("u")
    )
    source_vertices = (
        baryon_vertex("pbar_source", flavors=p, barred=True, external_spin=0),
        meson_vertex(
            "pi_minus_source", antiquark_flavor="u", quark_flavor="d", gamma=GAMMA5
        ),
    )
    return Correlator(
        Operator("p_pi_plus_sink", sink_fields),
        Operator.from_vertices("pbar_pi_minus_source", source_vertices),
    )


def make_pi_zero_connected() -> Correlator:
    sink_fields = (Field("u", barred=True), Field("u"))
    source = meson_vertex(
        "pi_zero_source", antiquark_flavor="u", quark_flavor="u", gamma=GAMMA5
    )
    return Correlator(Operator("pi_zero_sink", sink_fields),
                      Operator.from_vertices("pi_zero_source", (source,)))


__all__ = [
    "CG5",
    "GAMMA5",
    "IDENTITY",
    "baryon_vertex",
    "make_pnn",
    "make_p_pi",
    "make_pi_plus_pi_minus",
    "make_pi_zero_connected",
    "meson_vertex",
]
