"""Exact scalar and shared-basis linear-combination API for Wick reduction."""
from __future__ import annotations

import ast
import math
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Mapping, Optional, Sequence, Tuple

from kernels import correlator_wick_normalizer as core
from . import operator_model
from . import reducer as wick_reducer


def _squarefree(value: int) -> tuple[int, int]:
    outside, squarefree, prime = 1, 1, 2
    if value < 0:
        raise ValueError("sqrt radicands must be nonnegative")
    while prime * prime <= value:
        exponent = 0
        while value % prime == 0:
            value //= prime
            exponent += 1
        outside *= prime ** (exponent // 2)
        if exponent % 2:
            squarefree *= prime
        prime = 3 if prime == 2 else prime + 2
    if value > 1:
        squarefree *= value
    return outside, squarefree


@dataclass(frozen=True)
class ExactScalar:
    """Rational linear combination of square roots of positive rationals."""
    terms: Tuple[Tuple[int, Fraction], ...] = ((1, Fraction(1)),)

    def __post_init__(self) -> None:
        merged = {}
        for radicand, coefficient in self.terms:
            if not isinstance(radicand, int) or radicand < 1:
                raise ValueError("radicands must be positive integers")
            outside, root = _squarefree(radicand)
            merged[root] = merged.get(root, Fraction(0)) + Fraction(coefficient) * outside
        object.__setattr__(self, "terms", tuple((r, c) for r, c in sorted(merged.items()) if c))

    @classmethod
    def rational(cls, value: int | Fraction, denominator: int = 1) -> "ExactScalar":
        return cls(((1, Fraction(value) / denominator),))

    @classmethod
    def sqrt(cls, numerator: int, denominator: int = 1) -> "ExactScalar":
        value = Fraction(numerator, denominator)
        if value < 0:
            raise ValueError("real square root must be nonnegative")
        if not value:
            return cls(())
        return cls(((value.numerator * value.denominator,
                     Fraction(1, value.denominator)),))

    @classmethod
    def parse(cls, value: Any) -> "ExactScalar":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            raise TypeError("booleans are not coefficients")
        if isinstance(value, (int, Fraction)):
            return cls(((1, Fraction(value)),))
        if isinstance(value, str):
            return _parse_expression(value)
        if isinstance(value, Mapping) and "sqrt" in value:
            radicand = value["sqrt"]
            if isinstance(radicand, int):
                return cls.sqrt(radicand)
            if isinstance(radicand, (tuple, list)) and len(radicand) == 2:
                return cls.sqrt(int(radicand[0]), int(radicand[1]))
        raise TypeError(f"unsupported exact coefficient: {value!r}")

    def __add__(self, other: Any) -> "ExactScalar":
        return ExactScalar(self.terms + ExactScalar.parse(other).terms)

    __radd__ = __add__

    def __neg__(self) -> "ExactScalar":
        return ExactScalar(tuple((r, -c) for r, c in self.terms))

    def __sub__(self, other: Any) -> "ExactScalar":
        return self + (-ExactScalar.parse(other))

    def __rsub__(self, other: Any) -> "ExactScalar":
        return ExactScalar.parse(other) - self

    def __mul__(self, other: Any) -> "ExactScalar":
        right = ExactScalar.parse(other)
        return ExactScalar(tuple((a * b, x * y)
                                 for a, x in self.terms for b, y in right.terms))

    __rmul__ = __mul__

    def inverse(self) -> "ExactScalar":
        if len(self.terms) != 1:
            raise ZeroDivisionError("only a single radical term can be inverted")
        radicand, coefficient = self.terms[0]
        if not coefficient:
            raise ZeroDivisionError("division by zero")
        return ExactScalar(((radicand, Fraction(1, 1) / (coefficient * radicand)),))

    def __truediv__(self, other: Any) -> "ExactScalar":
        return self * ExactScalar.parse(other).inverse()

    def conjugate(self) -> "ExactScalar":
        return self  # supported exact scalars are real

    def to_float(self) -> float:
        return sum(float(c) * math.sqrt(r) for r, c in self.terms)

    def to_json(self) -> dict[str, Any]:
        return {"expression": str(self), "terms": [
            {"radicand": r, "coefficient": f"{c.numerator}/{c.denominator}"}
            for r, c in self.terms
        ]}

    def __bool__(self) -> bool:
        return bool(self.terms)

    def __str__(self) -> str:
        if not self.terms:
            return "0"
        pieces = []
        for root, coefficient in self.terms:
            value = abs(coefficient)
            factor = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
            body = factor if root == 1 else (f"sqrt({root})" if value == 1 else f"{factor}*sqrt({root})")
            if not pieces:
                pieces.append(("-" if coefficient < 0 else "") + body)
            else:
                pieces.append((" - " if coefficient < 0 else " + ") + body)
        return "".join(pieces)


def _parse_expression(expression: str) -> ExactScalar:
    tree = ast.parse(expression.strip(), mode="eval")
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return ExactScalar.rational(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return visit(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1:
            value = visit(node.args[0])
            if len(value.terms) != 1 or value.terms[0][0] != 1:
                raise ValueError("sqrt() argument must be rational")
            rational = value.terms[0][1]
            return ExactScalar.sqrt(rational.numerator, rational.denominator)
        raise ValueError(f"unsupported exact coefficient expression: {expression!r}")
    return visit(tree)


@dataclass(frozen=True)
class OperatorProductTerm:
    """One ordered operator product with an exact coefficient and origin label."""
    operators: Tuple[core.Operator, ...]
    coefficient: ExactScalar = ExactScalar()
    source_label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "operators", tuple(self.operators))
        object.__setattr__(self, "coefficient", ExactScalar.parse(self.coefficient))
        if not self.operators:
            raise ValueError("operator product cannot be empty")
        if len({operator.side for operator in self.operators}) != 1:
            raise ValueError("a product term must be entirely sink-side or source-side")

    @property
    def side(self) -> str:
        return self.operators[0].side


@dataclass(frozen=True)
class OperatorLinearCombination:
    """A sum of ordered operator products, with exact duplicate premerging."""
    terms: Tuple[OperatorProductTerm, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if not self.terms or len({term.side for term in self.terms}) != 1:
            raise ValueError("a combination needs nonempty terms from one side")

    @property
    def side(self) -> str:
        return self.terms[0].side

    def premerged(self) -> Tuple[OperatorProductTerm, ...]:
        merged = []
        for term in self.terms:
            found = next((item for item in merged if item[0] == term.operators), None)
            if found is None:
                merged.append([term.operators, term.coefficient, [term.source_label] if term.source_label else []])
            else:
                found[1] = found[1] + term.coefficient
                if term.source_label:
                    found[2].append(term.source_label)
        result = [OperatorProductTerm(ops, coeff, ", ".join(dict.fromkeys(labels)))
                  for ops, coeff, labels in merged if coeff]
        return tuple(sorted(result, key=lambda term: repr(term.operators)))


def lift_legacy_operators(operators: Sequence[core.Operator],
                           label: str = "correlator") -> tuple[OperatorLinearCombination, OperatorLinearCombination]:
    """Lift the legacy correlator to unit-weight sink and source combinations."""
    if len(operators) == 2 and all(isinstance(side, (tuple, list)) for side in operators):
        operators = tuple(operator for side in operators for operator in side)
    else:
        operators = tuple(operators)
    sink_ops = tuple(operator for operator in operators if operator.side == "sink")
    source_ops = tuple(operator for operator in operators if operator.side == "source")
    if not sink_ops or not source_ops:
        raise ValueError("legacy correlator must contain sink and source operators")
    return (
        OperatorLinearCombination((
            OperatorProductTerm(sink_ops, ExactScalar.rational(1), f"{label}:sink"),
        )),
        OperatorLinearCombination((
            OperatorProductTerm(source_ops, ExactScalar.rational(1), f"{label}:source"),
        )),
    )


def _complex_pair(value: complex) -> list[float]:
    value = complex(value)
    return [value.real, value.imag]


def reduce_linear_combination(
    sink_combination: OperatorLinearCombination,
    source_combination: OperatorLinearCombination,
    *,
    sink_name: str = "correlator",
    options: Optional[wick_reducer.ReductionOptions] = None,
    gamma_matrices: Optional[Mapping[str, core.Matrix]] = None,
    tolerance: float = wick_reducer.TOLERANCE,
) -> dict[str, Any]:
    """Expand components and assign one shared result-local basis ID space.

    Flavor-valid raw Wick pairings are generated from each original product.
    Their records are then passed together to the reducer with u/d flavor
    classes normalized, so transporter, VVV, open-axis and Dirac relations
    can map candidates from compatible source terms into shared basis states.
    Input coefficients remain exact; legacy Wick/Dirac relation multipliers
    are floating-point.  ``tolerance`` controls only the reported
    zero-within-tolerance flag, not the legacy reducer's fixed rank tolerance.
    """
    if not isinstance(sink_combination, OperatorLinearCombination):
        raise TypeError("sink_combination must be an OperatorLinearCombination")
    if not isinstance(source_combination, OperatorLinearCombination):
        raise TypeError("source_combination must be an OperatorLinearCombination")
    if sink_combination.side != "sink":
        raise ValueError("sink_combination must contain sink-side operators")
    if source_combination.side != "source":
        raise ValueError("source_combination must contain source-side operators")
    if not isinstance(sink_name, str) or not sink_name:
        raise ValueError("sink_name must be a nonempty string")
    if tolerance < 0 or not math.isfinite(float(tolerance)):
        raise ValueError("tolerance must be finite and nonnegative")
    isospin = True if options is None else options.isospin_degenerate
    source_terms = source_combination.premerged()
    sink_terms = sink_combination.premerged()
    batches: dict[Any, dict[str, Any]] = {}
    components = []
    started = time.perf_counter()

    def topology(operator):
        return (operator.side, operator.kind, operator.gamma,
                tuple(operator.gamma_slots), operator.open_slot,
                tuple("ud" if isospin and flavor.lower() in ("u", "d") else flavor
                      for flavor in operator.flavors),
                core.operator_factor_tag(operator), core.operator_factor_metadata(operator),
                operator.mode, operator_endpoint(operator))

    for sink_term in sink_terms:
        for source_term in source_terms:
            operators = sink_term.operators + source_term.operators
            # The source enters through its adjoint, so conjugate only its
            # coefficient here. Its field order and Gamma are supplied by
            # the source Operator definitions.
            exact_weight = sink_term.coefficient * source_term.coefficient.conjugate()
            fields, quarks, antiquarks = core._fields(operators)
            raw_terms = tuple(core.enumerate_wicks(quarks, antiquarks))
            component = {
                "sink": sink_name,
                "sink_source_label": sink_term.source_label,
                "source_label": source_term.source_label,
                "sink_input_coefficient": sink_term.coefficient.to_json(),
                "source_input_coefficient": source_term.coefficient.to_json(),
                "adjointed_source_coefficient": source_term.coefficient.conjugate().to_json(),
                "exact_product_weight": exact_weight.to_json(),
                "raw_wicks": [
                    {"local_wick_index": index, **wick_reducer._record_for_term(
                        index, term, fields, quarks, antiquarks,
                    )}
                    for index, term in enumerate(raw_terms, start=1)
                ],
                "wick_count": len(raw_terms),
            }
            components.append(component)
            if not raw_terms or not exact_weight:
                continue
            key = tuple(topology(operator) for operator in operators)
            batch = batches.get(key)
            if batch is None:
                normalized = tuple(replace(
                    operator,
                    flavors=tuple("ud" if isospin and flavor.lower() in ("u", "d") else flavor
                                  for flavor in operator.flavors),
                ) for operator in operators)
                batch = {"operators": normalized, "terms": [], "origins": []}
                batches[key] = batch
            for local_index, term in enumerate(raw_terms, start=1):
                global_index = len(batch["terms"]) + 1
                batch["terms"].append(term)
                batch["origins"].append({
                    "global_wick_index": global_index,
                    "component_index": len(components) - 1,
                    "sink": sink_name,
                    "sink_term_label": sink_term.source_label,
                    "source_label": source_term.source_label,
                    "exact_weight": exact_weight,
                    "term": term,
                    "local_wick_index": local_index,
                })

    basis_records: list[dict[str, Any]] = []
    trace_by_wick: dict[tuple[int, int], dict[str, Any]] = {}
    for batch in batches.values():
        operators = batch["operators"]
        origin_by_wick = {item["global_wick_index"]: item for item in batch["origins"]}
        captured_basis = []
        captured_relations = []

        def basis_capture(wick_index, term, state, aggregate, *_model_fields):
            captured_basis.append({
                "wick_index": wick_index, "term": term, "state": state,
                "unweighted_aggregate": complex(aggregate),
            })

        def relation_capture(relation, state, basis, term):
            captured_relations.append({
                "relation": dict(relation), "state": state, "term": term,
            })

        reduction = wick_reducer.reduce_operators(
            operators, label="linear_combination_global_batch", mode="detailed",
            options=options, gamma_matrices=gamma_matrices,
            _precomputed_wicks=tuple(batch["terms"]),
            _basis_callback=basis_capture,
            _relation_term_callback=relation_capture,
        )
        basis_id_by_wick = {}
        for entry in captured_basis:
            origin = origin_by_wick[entry["wick_index"]]
            basis_id = len(basis_records)
            basis_id_by_wick[entry["wick_index"]] = basis_id
            basis_records.append({
                "global_basis_id": basis_id,
                "batch_topology": repr(tuple(topology(operator) for operator in operators)),
                "representative_sink": origin["sink"],
                "representative_source_label": origin["source_label"],
                "representative_wick_index": entry["wick_index"],
                "representative_pairing": list(entry["term"].pairing),
                "representative_wick_sign": entry["term"].sign,
                "sink_vvv_sign": entry["state"].sink_vvv_sign,
                "source_vvv_sign": entry["state"].source_vvv_sign,
                "basis_state_coefficient": _complex_pair(entry["state"].coefficient),
                "unweighted_local_aggregate_numeric": _complex_pair(entry["unweighted_aggregate"]),
                "rows": {},
            })
            item = {
                "origin": origin, "term": entry["term"], "state": entry["state"],
                "relation": entry["state"].coefficient, "basis_id": basis_id,
            }
            _add_wick_trace(
                trace_by_wick, item, (basis_id, entry["state"].coefficient),
                residual_norm2=0.0,
            )
            _add_row_contribution(
                basis_records[basis_id], origin, 1 + 0j,
                basis_state_coefficient=entry["state"].coefficient,
                residual_norm2=0.0,
            )

        for entry in captured_relations:
            relation = entry["relation"]
            wick_index = int(relation["wick_index"])
            origin = origin_by_wick[wick_index]
            mappings = []
            for basis_wick, coefficient in zip(
                relation.get("basis_wick_indices", ()), relation.get("coefficients", ()),
            ):
                basis_id = basis_id_by_wick.get(int(basis_wick))
                if basis_id is None:
                    raise RuntimeError("dependent Wick points to a basis pivot not captured in its batch")
                value = complex(coefficient[0], coefficient[1])
                basis_state_coefficient = complex(
                    basis_records[basis_id]["basis_state_coefficient"][0],
                    basis_records[basis_id]["basis_state_coefficient"][1],
                )
                mappings.append((basis_id, value * basis_state_coefficient))
                _add_row_contribution(
                    basis_records[basis_id], origin, value,
                    basis_state_coefficient=basis_state_coefficient,
                    residual_norm2=float(relation.get("residual_norm2", math.inf)),
                )
            item = {"origin": origin, "term": entry["term"], "state": entry["state"]}
            _add_wick_trace(
                trace_by_wick, item, *mappings,
                residual_norm2=float(relation.get("residual_norm2", math.inf)),
            )

        batch["reduction"] = reduction

    # The reducer assigns local Wick indexes separately inside each compatible
    # topology batch.  Preserve that boundary explicitly in trace output.
    traces = sorted(trace_by_wick.values(), key=lambda item: (
        item["sink"], item["sink_term_label"], item["source_label"], item["local_wick_index"],
    ))
    public_basis = []
    for basis in basis_records:
        rows = {}
        row = basis["rows"].get(sink_name, {
            "numeric": 0j, "terms": [], "exact": ExactScalar.rational(0),
            "exact_safe": True,
        })
        value = row["numeric"]
        exact_value = row["exact"] if row["exact_safe"] else None
        rows[sink_name] = {
            "numeric": _complex_pair(value),
            "zero_within_tolerance": abs(value) <= tolerance,
            "exact_coefficient": exact_value.to_json() if exact_value is not None else None,
            "exact_zero": (not bool(exact_value)) if exact_value is not None else None,
            "coefficient_terms": row["terms"],
        }
        public_basis.append({key: value for key, value in basis.items() if key != "rows"} | {"rows": rows})

    return {
        "reduction_strategy": "expand_then_reduce_per_compatible_topology_batch",
        "shared_basis_within_batch": True,
        "shared_dirac_cache_within_batch": True,
        "output_emitted_after_batch": True,
        "sink_name": sink_name,
        "sink_names": [sink_name],
        "source_terms": [{"source_label": term.source_label,
                           "coefficient": term.coefficient.to_json(),
                           "operator_count": len(term.operators)} for term in source_terms],
        "source_normalization_mapping": {
            "statement": "The magnitude map is Sigma0_norm=Sigma0_raw/sqrt(2), giving 1/sqrt(3) on each normalized Sigma0_raw p and Sigma+ n_phys term. In this repository's raw baryon slot convention, I_+ n_raw=-p_raw, so n_phys=-n_raw and the explicit raw-operator coefficient of Sigma+_raw n_raw is -1/sqrt(3).",
            "normalized_physical_coefficients": [ExactScalar.sqrt(1, 3).to_json()] * 2,
            "raw_operator_coefficients": [ExactScalar.sqrt(1, 3).to_json(), (-ExactScalar.sqrt(1, 3)).to_json()],
            "neutron_phase_identity": "epsilon d_a (d_b^T Cgamma5 u_c) under I_+ equals -epsilon u_a (u_b^T Cgamma5 d_c); the other term has a uu Cgamma5 diquark and vanishes by epsilon antisymmetry.",
        },
        "wick_count": sum(component["wick_count"] for component in components),
        "global_basis_count": len(public_basis),
        "global_basis": public_basis,
        "wick_contributions": traces,
        "components": components,
        "global_batches": [{
            "topology": repr(tuple(topology(operator) for operator in batch["operators"])),
            "wick_count": batch["reduction"]["wick_count"],
            "independent_count": batch["reduction"]["independent_count"],
            "dependent_count": batch["reduction"]["dependent_count"],
            "peram_graph_bucket_count": batch["reduction"]["peram_incidence_buckets"],
        } for batch in batches.values()],
        "numeric_tolerance": wick_reducer.TOLERANCE,
        "zero_reporting_tolerance": tolerance,
        "global_independence_certified": False,
        "global_basis_scope": "result-local union of pivots from legacy peram-graph buckets; cross-bucket tensor rank is not tested",
        "exactness": {
            "input_coefficients": "exact rational and square-root algebra",
            "Wick_and_Dirac_relation_coefficients": "floating-point complex arithmetic",
            "global_rank_test": f"legacy Gram projection within each peram-graph bucket; fixed rank tolerance={wick_reducer.TOLERANCE:g}; cross-bucket complete-tensor rank is not tested",
            "exact_coefficient_zero": "the exact radical sum of input weights times accepted exactly integral real relation multipliers; relation discovery is still floating-point and is not a standalone symbolic proof",
            "global_basis_independence": "not certified across peram-graph buckets",
            "all_rows_exact_zero": all(
                row["exact_zero"] is True
                for basis in public_basis for row in basis["rows"].values()
            ),
        },
        "timing_seconds": {"total": time.perf_counter() - started},
    }


def _add_row_contribution(
    basis: dict[str, Any], origin: dict[str, Any], relation: complex,
    *, basis_state_coefficient: complex, residual_norm2: float,
) -> None:
    row = basis["rows"].setdefault(origin["sink"], {
        "numeric": 0j, "terms": [], "exact": ExactScalar.rational(0), "exact_safe": True,
    })
    complete_relation = relation * basis_state_coefficient
    value = origin["exact_weight"].to_float() * complete_relation
    row["numeric"] += value
    integral_real = (
        complete_relation.real == int(complete_relation.real)
        and complete_relation.imag == 0.0
    )
    if residual_norm2 == 0.0 and integral_real:
        row["exact"] = row["exact"] + origin["exact_weight"] * int(complete_relation.real)
    else:
        row["exact_safe"] = False
    row["terms"].append({
        "sink_term_label": origin["sink_term_label"],
        "source_label": origin["source_label"],
        "exact_input_weight": origin["exact_weight"].to_json(),
        "relation_numeric": _complex_pair(relation),
        "basis_state_coefficient": _complex_pair(basis_state_coefficient),
        "coefficient_in_unweighted_basis": _complex_pair(complete_relation),
        "weighted_relation_numeric": _complex_pair(value),
        "relation_residual_norm2": residual_norm2,
        "exact_integer_relation": residual_norm2 == 0.0 and integral_real,
    })


def _add_wick_trace(trace_by_wick: dict, item: dict[str, Any], *mapping,
                    residual_norm2: float) -> None:
    origin, term = item["origin"], item["term"]
    trace_by_wick[(origin["component_index"], origin["local_wick_index"])] = {
        "component_index": origin["component_index"],
        "sink": origin["sink"],
        "sink_term_label": origin["sink_term_label"],
        "source_label": origin["source_label"],
        "local_wick_index": origin["local_wick_index"],
        "pairing": list(term.pairing),
        "wick_sign": term.sign,
        "sink_vvv_sign": item["state"].sink_vvv_sign,
        "source_vvv_sign": item["state"].source_vvv_sign,
        "input_weight_exact": origin["exact_weight"].to_json(),
        "global_relation_residual_norm2": residual_norm2,
        "exact_integer_relation": residual_norm2 == 0.0 and all(
            coefficient.imag == 0.0 and coefficient.real == int(coefficient.real)
            for _basis_id, coefficient in mapping
        ),
        "global_basis_mapping": [
            {"global_basis_id": basis_id, "relation_numeric": _complex_pair(coefficient)}
            for basis_id, coefficient in mapping
        ],
    }


def operator_endpoint(operator: core.Operator):
    return core.operator_endpoint_descriptor(operator)

def make_sigma_lambda_example(
    sink_name: str = "Lambda1p", mode: str = "distillation",
) -> tuple[OperatorLinearCombination, OperatorLinearCombination]:
    """Build one Lambda-p sink combination and the common exact Sigma-N source.

    ``sink_name`` selects the single sink channel to calculate. Lambda1p and
    Lambda2p are intentionally separate calls in this input layer.
    """
    def baryon(side: str, species: str, label: str, flavors: Sequence[str]) -> core.Operator:
        gamma = "Cgamma5_dagger" if side == "source" else "Cgamma5"
        return operator_model.make_operator(
            side, "baryon", species, label, flavors, gamma, (1, 2), open_slot=0,
            mode=mode,
        )

    root = ExactScalar.sqrt(1, 3)
    sink_p = baryon("sink", "p", "sink_p", ("u", "u", "d"))
    lambda1 = baryon("sink", "Lambda1", "sink_Lambda1", ("s", "u", "d"))
    lambda2_first = baryon("sink", "Lambda2_dus", "sink_Lambda2_dus", ("d", "u", "s"))
    lambda2_second = baryon("sink", "Lambda2_uds", "sink_Lambda2_uds", ("u", "d", "s"))

    # Explicit adjoint convention: reverse the two baryon factors, reverse
    # each diquark slot pair, and use the daggered Gamma.  No source rewriting
    # is performed by the legacy Operator API itself.
    sigma0_dus_bar = baryon("source", "Sigma0_raw_dus", "source_Sigma0_dus", ("d", "s", "u"))
    sigma0_uds_bar = baryon("source", "Sigma0_raw_uds", "source_Sigma0_uds", ("u", "s", "d"))
    sigma_plus_bar = baryon("source", "Sigma_plus", "source_Sigma_plus", ("u", "s", "u"))
    p_bar = baryon("source", "pbar", "source_pbar", ("u", "d", "u"))
    n_bar = baryon("source", "nbar", "source_nbar", ("d", "u", "d"))

    source = OperatorLinearCombination((
        OperatorProductTerm((p_bar, sigma0_dus_bar), root, "Sigma0_raw:d(u^T Cgamma5 s) p"),
        OperatorProductTerm((p_bar, sigma0_uds_bar), root, "Sigma0_raw:u(d^T Cgamma5 s) p"),
        # With the project's raw nucleon convention B(u,u,d)=p and
        # B(d,d,u)=n, Cgamma5^T=-Cgamma5 plus epsilon antisymmetry gives
        # I_+ n_raw = -p_raw.  Thus the Condon-Shortley physical neutron is
        # -n_raw, and the physical plus-CG ket carries a minus on this raw term.
        OperatorProductTerm((n_bar, sigma_plus_bar), -root, "Sigma_plus n (raw n phase -1)"),
    ))
    sink1 = OperatorLinearCombination((
        OperatorProductTerm((lambda1, sink_p), ExactScalar.rational(1), "Lambda1 p"),
    ))
    sink2 = OperatorLinearCombination((
        OperatorProductTerm((lambda2_first, sink_p), ExactScalar.rational(1), "+ d(u^T Cgamma5 s) p"),
        OperatorProductTerm((lambda2_second, sink_p), ExactScalar.rational(-1), "- u(d^T Cgamma5 s) p"),
    ))
    if sink_name == "Lambda1p":
        return sink1, source
    if sink_name == "Lambda2p":
        return sink2, source
    raise ValueError("sink_name must be 'Lambda1p' or 'Lambda2p'")

