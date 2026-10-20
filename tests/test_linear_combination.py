"""Exact input weights and the flat four-step combination API."""

import math
import sys
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kernels.wick_reducer import (
    ExactScalar,
    OperatorLinearCombination,
    OperatorProductTerm,
    lift_legacy_operators,
    make_operator,
    make_sigma_lambda_example,
    reduce_linear_combination,
    reduce_operators,
)
from kernels.wick_reducer.operator_model import operator_library


def _combination(operators, coefficient=1, label="term"):
    return OperatorLinearCombination((OperatorProductTerm(
        tuple(operators), coefficient, label,
    ),))


def _basis_coefficients(report):
    return [
        basis["rows"][report["sink_name"]]["exact_coefficient"]["expression"]
        for basis in report["global_basis"]
    ]


class ExactScalarTests(unittest.TestCase):
    def test_rational_and_clebsch_gordan_roots_stay_exact(self):
        root = ExactScalar.parse("sqrt(2/3)")
        self.assertEqual(root.to_json()["terms"], [
            {"radicand": 6, "coefficient": "1/3"},
        ])
        self.assertEqual(
            (ExactScalar.parse("sqrt(1/3)") * 3).to_json()["expression"],
            "sqrt(3)",
        )
        self.assertEqual(ExactScalar.parse(Fraction(2, 3)).to_json()["expression"], "2/3")
        with self.assertRaises((TypeError, ValueError)):
            ExactScalar.parse(0.5)


class LinearCombinationReductionTests(unittest.TestCase):
    def test_operator_label_is_optional_but_flavors_are_required(self):
        operator = make_operator(
            side="sink", kind="baryon", species="p",
            flavors=("u", "u", "d"), gamma="Cgamma5",
            gamma_slots=(1, 2), open_slot=0,
        )
        self.assertEqual(operator.label, "sink_p")
        with self.assertRaisesRegex(ValueError, "flavors are required"):
            make_operator(side="sink", kind="baryon", species="p")

    def test_flat_four_step_api_accepts_same_operator_twice_for_pp(self):
        p_sink, p_source = operator_library()["ppn"][0], operator_library()["ppn"][3]
        sink = _combination((p_sink, p_sink), label="pp sink")
        source = _combination((p_source, p_source), label="pp source")
        result = reduce_linear_combination(sink, source, sink_name="pp")
        self.assertEqual(result["sink_names"], ["pp"])
        self.assertEqual(result["wick_count"], 48)
        self.assertEqual(result["global_basis_count"], 15)

    def test_expanded_terms_share_one_reducer_call_per_topology_batch(self):
        sink, source = operator_library()["delta"]
        sink_combination = _combination((sink,), label="sink")
        source_terms = (
            OperatorProductTerm((source,), Fraction(1, 3), "source-1"),
            OperatorProductTerm((replace(source, label="source-2"),), Fraction(2, 3), "source-2"),
        )
        source_combination = OperatorLinearCombination(source_terms)
        original = reduce_operators
        calls = []

        def counted(operators, **kwargs):
            calls.append(tuple(operators))
            return original(operators, **kwargs)

        with patch("kernels.wick_reducer.linear_combination.wick_reducer.reduce_operators", counted):
            result = reduce_linear_combination(sink_combination, source_combination, sink_name="delta")
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["global_batches"][0]["wick_count"], 12)

    def test_distribution_over_distinct_source_terms(self):
        sink, source_g3 = operator_library()["delta"]
        source_g5 = replace(
            source_g3, label="source_delta_gamma5", gamma="Cgamma5_dagger",
        )
        sink_combination = _combination((sink,), label="Delta++ sink")
        source_terms = (
            OperatorProductTerm((source_g3,), Fraction(1, 3), "source gamma3"),
            OperatorProductTerm((source_g5,), Fraction(2, 3), "source gamma5"),
        )
        source_combination = OperatorLinearCombination(source_terms)
        together = reduce_linear_combination(sink_combination, source_combination, sink_name="delta")
        first = reduce_linear_combination(
            sink_combination, OperatorLinearCombination((source_terms[0],)), sink_name="delta",
        )
        second = reduce_linear_combination(
            sink_combination, OperatorLinearCombination((source_terms[1],)), sink_name="delta",
        )
        self.assertEqual(len(together["global_batches"]), 2)

        def coefficient_map(report):
            return {
                (basis["batch_topology"], basis["representative_wick_index"],
                 tuple(basis["representative_pairing"])):
                    ExactScalar.parse(basis["rows"]["delta"]["exact_coefficient"]["expression"])
                for basis in report["global_basis"]
            }

        together_map = coefficient_map(together)
        expected = {}
        for report in (first, second):
            expected.update(coefficient_map(report))
        self.assertEqual(together_map, expected)

    def test_legacy_single_product_matches_old_api_coefficients(self):
        operators = operator_library()["pn"]
        old = reduce_operators(operators, mode="detailed")
        sink_combination, source_combination = lift_legacy_operators(operators)
        new = reduce_linear_combination(sink_combination, source_combination)

        self.assertEqual(new["wick_count"], old["wick_count"])
        self.assertEqual(new["global_basis_count"], old["independent_count"])
        old_coefficients = {
            item["wick_index"]: item["coefficient"]
            for item in old["independent_wicks"]
        }
        new_coefficients = {
            basis["representative_wick_index"]:
                basis["rows"]["correlator"]["numeric"]
            for basis in new["global_basis"]
        }
        self.assertEqual(old_coefficients.keys(), new_coefficients.keys())
        for wick_index, value in old_coefficients.items():
            self.assertEqual(new_coefficients[wick_index], value)

    def test_distribution_premerge_reordering_and_exact_scaling(self):
        operators = operator_library()["delta"]
        sink_ops = tuple(op for op in operators if op.side == "sink")
        source_ops = tuple(op for op in operators if op.side == "source")
        sink = _combination(sink_ops, label="unit sink")
        source = _combination(source_ops, label="unit source")
        split_source = OperatorLinearCombination((
            OperatorProductTerm(source_ops, Fraction(1, 3), "source / 3"),
            OperatorProductTerm(source_ops, Fraction(2, 3), "2 source / 3"),
        ))

        unit_result = reduce_linear_combination(sink, source, sink_name="delta")
        split_result = reduce_linear_combination(sink, split_source, sink_name="delta")
        reversed_result = reduce_linear_combination(
            sink, OperatorLinearCombination(tuple(reversed(split_source.terms))), sink_name="delta",
        )
        self.assertEqual(len(split_source.premerged()), 1)
        self.assertEqual(_basis_coefficients(unit_result), _basis_coefficients(split_result))
        self.assertEqual(_basis_coefficients(split_result), _basis_coefficients(reversed_result))

        scaled = OperatorLinearCombination((
            OperatorProductTerm(sink_ops, "sqrt(2)", "sqrt2 sink"),
        ))
        scaled_result = reduce_linear_combination(scaled, source, sink_name="delta")
        self.assertEqual(scaled_result["global_basis_count"], unit_result["global_basis_count"])
        for original, changed in zip(unit_result["global_basis"], scaled_result["global_basis"]):
            a = original["rows"]["delta"]["numeric"][0]
            b = changed["rows"]["delta"]["numeric"][0]
            self.assertTrue(math.isclose(b, math.sqrt(2) * a, rel_tol=1e-13, abs_tol=1e-13))

    def test_lambda1_and_lambda2_are_separate_single_sink_requests(self):
        for name, wick_count, basis_count in (
            ("Lambda1p", 36, 18), ("Lambda2p", 72, 33),
        ):
            sink, source = make_sigma_lambda_example(name)
            report = reduce_linear_combination(sink, source, sink_name=name)
            self.assertEqual(report["sink_names"], [name])
            self.assertEqual(report["wick_count"], wick_count)
            self.assertEqual(report["global_basis_count"], basis_count)
            self.assertTrue(report["exactness"]["all_rows_exact_zero"])
            for basis in report["global_basis"]:
                self.assertTrue(basis["rows"][name]["exact_zero"])
                self.assertTrue(basis["rows"][name]["zero_within_tolerance"])

    def test_source_term_order_invariance_and_raw_neutron_phase_matters(self):
        sink, source = make_sigma_lambda_example("Lambda1p")
        shuffled = OperatorLinearCombination(tuple(reversed(source.terms)))
        first = reduce_linear_combination(sink, source, sink_name="Lambda1p")
        second = reduce_linear_combination(sink, shuffled, sink_name="Lambda1p")
        self.assertEqual(_basis_coefficients(first), _basis_coefficients(second))

        wrong_phase_terms = list(source.terms)
        wrong_phase_terms[-1] = replace(
            wrong_phase_terms[-1], coefficient=-wrong_phase_terms[-1].coefficient,
        )
        wrong_report = reduce_linear_combination(
            sink, OperatorLinearCombination(tuple(wrong_phase_terms)), sink_name="Lambda1p",
        )
        self.assertTrue(any(
            basis["rows"]["Lambda1p"]["exact_zero"] is False
            for basis in wrong_report["global_basis"]
        ))


if __name__ == "__main__":
    unittest.main()
