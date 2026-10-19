"""Streaming, single-file output for exact ordinary-colour reductions."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import os
import sqlite3
import sys
import tempfile
import time
from io import StringIO
from typing import Mapping, Sequence

from . import exact_operator_backend as exact
from .user_api import HadronOperator, _adapt_operator_pair, normalize_correlator


def _number(value: complex):
    number = complex(value)
    return [number.real, number.imag]


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _operator_metadata(operator: HadronOperator):
    return {
        "side": operator.side,
        "kind": operator.kind,
        "species": operator.species,
        "label": operator.label,
        "flavors": list(operator.flavors),
        "gamma": operator.gamma,
        "gamma_slots": list(operator.gamma_slots),
        "open_slot": operator.open_slot,
        "external_spin": operator.external_spin,
        "external_tag": operator.external_tag,
        "point": operator.point,
    }


def _explicit_metadata(operator: exact.Operator, side: str):
    metadata = {
        "side": side,
        "kind": "explicit_sparse_tensor",
        "label": operator.name,
        "fields": [
            {"flavor": field.flavor, "barred": field.barred, "point": field.point}
            for field in operator.fields
        ],
        "open_dimensions": list(operator.open_dimensions),
    }
    if operator.factors:
        metadata["factors"] = [
            _explicit_factor_metadata(factor) for factor in operator.factors
        ]
    else:
        metadata["terms"] = [
            {
                "coordinates": [list(coordinate) for coordinate in term.coordinates],
                "coefficient": _number(term.coefficient),
                "open_indices": list(term.open_indices),
            }
            for term in operator.terms
        ]
    return metadata


def _explicit_factor_metadata(operator: exact.Operator):
    metadata = {
        "label": operator.name,
        "fields": [
            {"flavor": field.flavor, "barred": field.barred, "point": field.point}
            for field in operator.fields
        ],
        "open_dimensions": list(operator.open_dimensions),
    }
    if operator.factors:
        metadata["factors"] = [
            _explicit_factor_metadata(factor) for factor in operator.factors
        ]
    else:
        metadata["terms"] = [
            {
                "coordinates": [list(coordinate) for coordinate in term.coordinates],
                "coefficient": _number(term.coefficient),
                "open_indices": list(term.open_indices),
            }
            for term in operator.terms
        ]
    return metadata


def _field_order(operators):
    result = []
    index = 0
    for side_name, side in zip(("sink", "source"), operators):
        for operator_index, operator in enumerate(side):
            if isinstance(operator, HadronOperator):
                fields = tuple(
                    exact.Field(
                        flavor,
                        operator.kind == "meson" and slot == 0
                        or operator.kind == "baryon" and side_name == "source",
                        operator.point,
                    )
                    for slot, flavor in enumerate(operator.flavors)
                )
                label = operator.label
            else:
                fields = operator.fields
                label = operator.name
            for slot, field in enumerate(fields):
                result.append({
                    "field_index": index,
                    "side": side_name,
                    "operator_index": operator_index,
                    "operator_label": label,
                    "slot": slot,
                    "flavor": field.flavor,
                    "barred": field.barred,
                    "point": field.point,
                })
                index += 1
    return result


def _profile(profile):
    return [
        {"flavor": flavor, "quark_point": q_point,
         "antiquark_point": anti_point, "count": count}
        for ((flavor, q_point, anti_point), count) in profile
    ]


def _typed_profile(profile):
    if profile is None:
        return None
    return {
        "connectivity_class": list(profile.connectivity_class),
        "topology_signature": list(profile.topology_signature),
        "edges": [
            {
                "flavor": flavor,
                "quark": {
                    "field_index": quark[0], "side": quark[1],
                    "operator_index": quark[2], "slot": quark[3],
                    "point": quark[4], "tensor_role": quark[5],
                },
                "antiquark": {
                    "field_index": antiquark[0], "side": antiquark[1],
                    "operator_index": antiquark[2], "slot": antiquark[3],
                    "point": antiquark[4], "tensor_role": antiquark[5],
                },
            }
            for flavor, quark, antiquark in profile.edges
        ],
        "output_route": [
            {"axis": axis, "side": side, "operator_index": operator,
             "local_axis": local_axis, "dimension": dimension}
            for axis, side, operator, local_axis, dimension in profile.output_route
        ],
    }


def _transporter(transporter):
    if transporter is None:
        return None
    return {
        "representative_wick_index": transporter.representative_wick_index,
        "field_map": list(transporter.field_map),
        "open_axis_map": list(transporter.open_axis_map),
        "tensor_phase": _number(transporter.tensor_phase),
        "wick_sign_ratio": _number(transporter.wick_sign_ratio),
        "phase": _number(transporter.phase),
    }


def _pairing(pairing):
    return {"pairs": [list(pair) for pair in pairing.pairs], "sign": pairing.sign}


def _vector_digest(vector: Mapping, *, tolerance: float) -> str:
    digest = sha256()
    for key, coefficient in sorted(vector.items(), key=lambda item: item[0]):
        if abs(coefficient) <= tolerance:
            continue
        digest.update(_json([key[0], key[1]]).encode("utf-8"))
        digest.update(b"=")
        digest.update(_json(_number(coefficient)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_basis_record(stream, basis_entry, *, wick_index: int):
    vector = basis_entry.vector
    stream.write("{\"type\":\"basis\",\"basis_index\":")
    stream.write(str(basis_entry.index))
    stream.write(",\"wick_index\":")
    stream.write(str(wick_index))
    stream.write(",\"pairing\":")
    stream.write(_json(_pairing(basis_entry.pairing)))
    stream.write(",\"propagator_profile\":")
    stream.write(_json(_profile(basis_entry.propagator_profile)))
    stream.write(",\"typed_profile\":")
    stream.write(_json(_typed_profile(basis_entry.typed_profile)))
    stream.write(",\"connectivity_class\":")
    stream.write(_json(list(basis_entry.typed_profile.connectivity_class)))
    stream.write(",\"term_count\":")
    stream.write(str(len(vector)))
    stream.write(",\"terms\":[")
    first = True
    for (open_indices, monomial), coefficient in sorted(vector.items(), key=lambda item: item[0]):
        if not first:
            stream.write(",")
        first = False
        stream.write(_json({
            "open_indices": list(open_indices),
            "peram_elements": [
                {"flavor": factor[0], "quark_point": factor[1],
                 "quark_color": factor[2], "quark_spin": factor[3],
                 "antiquark_point": factor[4], "antiquark_color": factor[5],
                 "antiquark_spin": factor[6]}
                for factor in monomial
            ],
            "coefficient": _number(coefficient),
        }))
    stream.write("]}")


def _write_relation_record(stream, relation, vector, *, tolerance: float):
    stream.write(_json({
        "type": "raw_wick_relation",
        "wick_index": relation.wick_index,
        "connectivity_class": list(relation.typed_profile.connectivity_class),
        "topology_signature": list(relation.typed_profile.topology_signature),
        "wick_sign": relation.pairing.sign,
        "pairing": _pairing(relation.pairing),
        "propagator_profile": _profile(relation.propagator_profile),
        "typed_profile": _typed_profile(relation.typed_profile),
        "transporter": _transporter(relation.transporter),
        "raw_term_count": len(vector),
        "raw_vector_sha256": _vector_digest(vector, tolerance=tolerance),
        "basis_terms": [
            {"basis_index": basis_index, "coefficient": _number(coefficient)}
            for basis_index, coefficient in relation.basis_coefficients
        ],
    }))


def write_output(
    operators: Sequence[Sequence[HadronOperator | exact.Operator]],
    path: str | Path,
    *,
    name: str = "correlator",
    mode: str = "independent",
    normal_ordered: bool = False,
    wick_scope: str = "complete",
    isospin_degenerate_perams: bool = True,
    tolerance: float = 1.0e-12,
    gamma_matrices: Mapping[str, Sequence[Sequence[complex]]] | None = None,
) -> dict[str, int | str]:
    """Write exact independent/check output as one streaming JSON document.

    The JSON has an ordered ``records`` array. A ``basis`` record contains
    the explicit coefficient monomials for one independent Wick; in check
    mode, each raw Wick also gets a ``raw_wick_relation`` record. The latter
    reconstructs the raw coefficient vector as a linear combination of basis
    vectors and includes a SHA-256 digest for downstream verification.
    """
    if mode not in {"independent", "check"}:
        raise ValueError("mode must be 'independent' or 'check'")
    if wick_scope not in {"complete", "cross_time"}:
        raise ValueError("wick_scope must be 'complete' or 'cross_time'")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if type(isospin_degenerate_perams) is not bool:
        raise ValueError("isospin_degenerate_perams must be true or false")
    operators = _adapt_operator_pair(operators)
    correlator, registry = normalize_correlator(
        operators, gamma_matrices=gamma_matrices
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    raw_wicks = basis_count = basis_terms = transporter_relations = 0
    group_stats = {}
    try:
        with tempfile.TemporaryDirectory(
            prefix="wick-output-groups-", dir=output_path.parent,
        ) as work_directory:
            connection = sqlite3.connect(Path(work_directory) / "records.sqlite3")
            connection.execute(
                "CREATE TABLE records (class_sin INTEGER, class_sou INTEGER, "
                "sequence INTEGER, record TEXT NOT NULL, "
                "PRIMARY KEY(class_sin, class_sou, sequence)) WITHOUT ROWID"
            )
            sequence = 0

            def spool_record(signature, writer, *args, **kwargs):
                nonlocal sequence
                buffer = StringIO()
                writer(buffer, *args, **kwargs)
                connection.execute(
                    "INSERT INTO records VALUES (?, ?, ?, ?)",
                    (*signature, sequence, buffer.getvalue()),
                )
                sequence += 1

            for basis_entry, relation in exact.iter_reduced_wicks(
                correlator,
                normal_ordered=normal_ordered,
                wick_scope=wick_scope,
                tolerance=tolerance,
                isospin_degenerate_perams=isospin_degenerate_perams,
            ):
                signature = relation.typed_profile.connectivity_class
                counters = group_stats.setdefault(signature, {
                    "raw_wicks": 0,
                    "independent_wicks": 0,
                    "basis_terms": 0,
                    "transporter_relations": 0,
                })
                raw_wicks += 1
                counters["raw_wicks"] += 1
                if (relation.transporter is not None
                        and relation.transporter.representative_wick_index
                        != relation.wick_index):
                    transporter_relations += 1
                    counters["transporter_relations"] += 1
                if basis_entry is not None:
                    spool_record(
                        signature, _write_basis_record, basis_entry,
                        wick_index=relation.wick_index,
                    )
                    basis_count += 1
                    basis_terms += len(basis_entry.vector)
                    counters["independent_wicks"] += 1
                    counters["basis_terms"] += len(basis_entry.vector)
                if mode == "check":
                    relation_vector = (
                        basis_entry.vector if basis_entry is not None
                        else relation.raw_vector if relation.raw_vector is not None
                        else _raw_vector_for_relation(
                            correlator, relation,
                            normal_ordered=normal_ordered,
                            isospin_degenerate_perams=isospin_degenerate_perams,
                            tolerance=tolerance,
                        )
                    )
                    spool_record(
                        signature, _write_relation_record, relation,
                        relation_vector, tolerance=tolerance,
                    )
            connection.commit()

            ordered_signatures = sorted(
                group_stats, key=lambda value: (value != (0, 0), value),
            )
            group_summaries = [
                {
                    "signature": list(signature),
                    "label": "pure_cross_time" if signature == (0, 0)
                    else "same_side_pairings",
                    **group_stats[signature],
                    "dependent_wicks": (
                        group_stats[signature]["raw_wicks"]
                        - group_stats[signature]["independent_wicks"]
                    ),
                }
                for signature in ordered_signatures
            ]
            summary = {
                "raw_wicks": raw_wicks,
                "independent_wicks": basis_count,
                "dependent_wicks": raw_wicks - basis_count,
                "basis_terms": basis_terms,
                "transporter_relations": transporter_relations,
                "groups": group_summaries,
            }
            try:
                with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write("{\n\"format\":\"wick-colour-orbit-json-v3\",\n")
                    stream.write("\"name\":")
                    stream.write(_json(name))
                    stream.write(",\n\"mode\":")
                    stream.write(_json(mode))
                    stream.write(",\n\"normal_ordered\":")
                    stream.write("true" if normal_ordered else "false")
                    stream.write(",\n\"wick_scope\":")
                    stream.write(_json(wick_scope))
                    stream.write(",\n\"isospin_degenerate_perams\":")
                    stream.write("true" if isospin_degenerate_perams else "false")
                    stream.write(",\n\"tolerance\":")
                    stream.write(repr(tolerance))
                    stream.write(",\n\"gamma_matrices\":")
                    gamma_names = sorted({operator.gamma for side in operators
                                          for operator in side
                                          if isinstance(operator, HadronOperator)
                                          and operator.gamma})
                    stream.write(_json({
                        matrix_name: [
                            [[complex(value).real, complex(value).imag] for value in row]
                            for row in registry[matrix_name]
                        ]
                        for matrix_name in gamma_names
                    }))
                    stream.write(",\n\"operators\":")
                    stream.write(_json([
                        [_operator_metadata(operator) if isinstance(operator, HadronOperator)
                         else _explicit_metadata(operator, side_name)
                         for operator in side]
                        for side_name, side in zip(("sink", "source"), operators)
                    ]))
                    stream.write(",\n\"field_order\":")
                    stream.write(_json(_field_order(operators)))
                    stream.write(",\n\"groups\":")
                    stream.write(_json(group_summaries))
                    stream.write(",\n\"records\":[\n")
                    first_record = True
                    cursor = connection.execute(
                        "SELECT record FROM records ORDER BY "
                        "CASE WHEN class_sin = 0 AND class_sou = 0 THEN 0 ELSE 1 END, "
                        "class_sin, class_sou, sequence"
                    )
                    for (record,) in cursor:
                        if not first_record:
                            stream.write(",\n")
                        stream.write(record)
                        first_record = False
                    if not first_record:
                        stream.write(",\n")
                    stream.write(_json({"type": "summary", **summary}))
                    stream.write("\n],\n\"summary\":")
                    stream.write(_json(summary))
                    stream.write("\n}\n")
                os.replace(temporary_path, output_path)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
            finally:
                connection.close()
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return {**summary, "output_bytes": output_path.stat().st_size}


def write_count_output(
    operators: Sequence[Sequence[HadronOperator | exact.Operator]],
    path: str | Path,
    *,
    name: str = "correlator",
    normal_ordered: bool = False,
    wick_scope: str = "complete",
    isospin_degenerate_perams: bool = True,
    tolerance: float = 1.0e-12,
    gamma_matrices: Mapping[str, Sequence[Sequence[complex]]] | None = None,
    batch_size: int = 4096,
) -> dict[str, int | float | str | None]:
    """Write a summary after exact coordinates or validated orbit reps merge."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if wick_scope not in {"complete", "cross_time"}:
        raise ValueError("wick_scope must be 'complete' or 'cross_time'")
    if type(isospin_degenerate_perams) is not bool:
        raise ValueError("isospin_degenerate_perams must be true or false")
    operators = _adapt_operator_pair(operators)
    correlator, registry = normalize_correlator(
        operators, gamma_matrices=gamma_matrices
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="wick-colour-count-", dir=output_path.parent,
    ) as work_directory:
        summary = exact.count_colour_orbit_sqlite(
            correlator,
            Path(work_directory) / "exact_terms.sqlite3",
            normal_ordered=normal_ordered,
            wick_scope=wick_scope,
            isospin_degenerate_perams=isospin_degenerate_perams,
            batch_size=batch_size,
            zero_tolerance=tolerance,
        )
    summary["runtime_seconds"] = time.perf_counter() - start
    summary["peak_rss_bytes"] = _peak_rss_bytes()
    summary["ncontr"] = None

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("{\n\"format\":\"wick-colour-orbit-count-v3\",\n")
            stream.write("\"name\":")
            stream.write(_json(name))
            stream.write(",\n\"mode\":\"count\",\n")
            stream.write("\"normal_ordered\":")
            stream.write("true" if normal_ordered else "false")
            stream.write(",\n\"wick_scope\":")
            stream.write(_json(wick_scope))
            stream.write(",\n\"isospin_degenerate_perams\":")
            stream.write("true" if isospin_degenerate_perams else "false")
            stream.write(",\n\"zero_tolerance\":")
            stream.write(repr(tolerance))
            gamma_names = sorted({operator.gamma for side in operators
                                  for operator in side
                                  if isinstance(operator, HadronOperator)
                                  and operator.gamma})
            stream.write(",\n\"gamma_matrices\":")
            stream.write(_json({
                matrix_name: [
                    [[complex(value).real, complex(value).imag] for value in row]
                    for row in registry[matrix_name]
                ]
                for matrix_name in gamma_names
            }))
            stream.write(",\n\"operators\":")
            stream.write(_json([
                [_operator_metadata(operator) if isinstance(operator, HadronOperator)
                 else _explicit_metadata(operator, side_name)
                 for operator in side]
                for side_name, side in zip(("sink", "source"), operators)
            ]))
            stream.write(",\n\"field_order\":")
            stream.write(_json(_field_order(operators)))
            stream.write(",\n\"summary\":")
            stream.write(_json(summary))
            stream.write("\n}\n")
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    summary["output_bytes"] = output_path.stat().st_size
    return summary


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _raw_vector_for_relation(
    correlator,
    relation,
    *,
    normal_ordered,
    isospin_degenerate_perams,
    tolerance,
):
    """Rebuild only a dependent raw vector for its check digest."""
    from .exact_operator_backend import _flatten, _iter_pairing_terms

    fields = _flatten(correlator)
    vector = {}
    for key, value in _iter_pairing_terms(
        correlator,
        fields,
        relation.pairing,
        isospin_degenerate_perams=isospin_degenerate_perams,
    ):
        vector[key] = vector.get(key, 0j) + value
    scale = max((abs(value) for value in vector.values()), default=0.0)
    threshold = tolerance * scale
    return {key: value for key, value in vector.items() if abs(value) > threshold}


__all__ = ["write_output", "write_count_output"]
