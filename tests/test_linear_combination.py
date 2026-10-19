"""Exact input weights and multi-component Wick-reduction regressions."""

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
    CorrelatorCombination,
    ExactScalar,
    OperatorLinearCombination,
    OperatorProductTerm,
    SinkOperator,
    SourceLinearCombination,
    lift_legacy_operators,
    make_sigma_lambda_example,
    reduce_linear_combination,
    reduce_operators,
)
from kernels.wick_reducer.operator_model import operator_library


def _basis_coefficients(report, sink_name):
    return [
        basis["rows"][sink_name]["exact_coefficient"]["expression"]
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
        self.assertEqual(ExactScalar.rational(Fraction(3, 4), 3).to_json()["expression"], "1/4")
        with self.assertRaises((TypeError, ValueError)):
            ExactScalar.parse(0.5)


class LinearCombinationReductionTests(unittest.TestCase):
    def test_expanded_terms_share_one_reducer_call_per_topology_batch(self):
        sink, source = operator_library()["delta"]
        request = CorrelatorCombination(
            (SinkOperator("delta", OperatorLinearCombination((
                OperatorProductTerm((sink,), 1, "sink"),
            ))),),
            SourceLinearCombination(OperatorLinearCombination((
                OperatorProductTerm((source,), Fraction(1, 3), "source-1"),
                OperatorProductTerm((replace(source, label="source-2"),), Fraction(2, 3), "source-2"),
            ))),
        )
        original = reduce_operators
        calls = []

        def counted(operators, **kwargs):
            calls.append(tuple(operators))
            return original(operators, **kwargs)

        with patch("kernels.wick_reducer.linear_combination.wick_reducer.reduce_operators", counted):
            result = reduce_linear_combination(request)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["global_batches"][0]["wick_count"], 12)

    def test_distribution_over_distinct_source_terms(self):
        sink, source_g3 = operator_library()["delta"]
        source_g5 = replace(
            source_g3, label="source_delta_gamma5", gamma="Cgamma5_dagger",
        )
        sink_combination = OperatorLinearCombination((
            OperatorProductTerm((sink,), 1, "Delta++ sink"),
        ))
        source_terms = (
            OperatorProductTerm((source_g3,), Fraction(1, 3), "source gamma3"),
            OperatorProductTerm((source_g5,), Fraction(2, 3), "source gamma5"),
        )
        combined = CorrelatorCombination(
            (SinkOperator("delta", sink_combination),),
            SourceLinearCombination(OperatorLinearCombination(source_terms)),
        )

        def one_term(term):
            return CorrelatorCombination(
                combined.sinks,
                SourceLinearCombination(OperatorLinearCombination((term,))),
            )

        together = reduce_linear_combination(combined)
        first = reduce_linear_combination(one_term(source_terms[0]))
        second = reduce_linear_combination(one_term(source_terms[1]))
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
        new = reduce_linear_combination(lift_legacy_operators(operators))

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
        unit = CorrelatorCombination(
            (SinkOperator("delta", OperatorLinearCombination((
                OperatorProductTerm(sink_ops, 1, "unit sink"),
            ))),),
            SourceLinearCombination(OperatorLinearCombination((
                OperatorProductTerm(source_ops, 1, "unit source"),
            ))),
        )
        split_source = OperatorLinearCombination((
            OperatorProductTerm(source_ops, Fraction(1, 3), "source / 3"),
            OperatorProductTerm(source_ops, Fraction(2, 3), "2 source / 3"),
        ))
        split = replace(unit, source=SourceLinearCombination(split_source))
        reversed_split = replace(
            unit,
            source=SourceLinearCombination(OperatorLinearCombination(tuple(reversed(split_source.terms)))),
        )

        unit_result = reduce_linear_combination(unit)
        split_result = reduce_linear_combination(split)
        reversed_result = reduce_linear_combination(reversed_split)
        self.assertEqual(len(split_source.premerged()), 1)
        self.assertEqual(
            _basis_coefficients(unit_result, "delta"),
            _basis_coefficients(split_result, "delta"),
        )
        self.assertEqual(
            _basis_coefficients(split_result, "delta"),
            _basis_coefficients(reversed_result, "delta"),
        )

        scaled = replace(
            unit,
            sinks=(SinkOperator("delta", OperatorLinearCombination((
                OperatorProductTerm(sink_ops, "sqrt(2)", "sqrt2 sink"),
            ))),),
        )
        scaled_result = reduce_linear_combination(scaled)
        self.assertEqual(scaled_result["global_basis_count"], unit_result["global_basis_count"])
        for original, changed in zip(
            unit_result["global_basis"], scaled_result["global_basis"],
        ):
            a = original["rows"]["delta"]["numeric"][0]
            b = changed["rows"]["delta"]["numeric"][0]
            self.assertTrue(math.isclose(b, math.sqrt(2) * a, rel_tol=1e-13, abs_tol=1e-13))
            exact = changed["rows"]["delta"]["exact_coefficient"]["expression"]
            self.assertNotIn("e-", exact)

    def test_sigma_n_to_lambda_p_exact_isospin_cancellation_and_trace(self):
        request = make_sigma_lambda_example()
        source_terms = request.source.combination.premerged()
        coefficient_by_label = {
            term.source_label: term.coefficient.to_json()["expression"]
            for term in source_terms
        }
        self.assertEqual(
            sorted(coefficient_by_label.values()),
            ["-1/3*sqrt(3)", "1/3*sqrt(3)", "1/3*sqrt(3)"],
        )
        self.assertEqual(
            coefficient_by_label["Sigma_plus n (raw n phase -1)"],
            "-1/3*sqrt(3)",
        )

        report = reduce_linear_combination(request)
        self.assertEqual(report["sink_names"], ["Lambda1p", "Lambda2p"])
        self.assertEqual(report["wick_count"], 108)
        self.assertEqual(report["global_basis_count"], 51)
        self.assertEqual(len(report["wick_contributions"]), 108)
        self.assertEqual(report["source_normalization_mapping"]["raw_operator_coefficients"], [
            ExactScalar.sqrt(1, 3).to_json(), (-ExactScalar.sqrt(1, 3)).to_json(),
        ])
        for sink in report["sink_names"]:
            for basis in report["global_basis"]:
                row = basis["rows"][sink]
                self.assertTrue(row["exact_zero"])
                self.assertTrue(row["zero_within_tolerance"])
                self.assertEqual(row["exact_coefficient"]["expression"], "0")
        for wick in report["wick_contributions"]:
            self.assertIn(wick["wick_sign"], (-1, 1))
            self.assertIn("sink_vvv_sign", wick)
            self.assertIn("source_vvv_sign", wick)
            self.assertIn("input_weight_exact", wick)
            self.assertTrue(wick["global_basis_mapping"])
            self.assertLessEqual(
                max(mapping["global_basis_id"] for mapping in wick["global_basis_mapping"]),
                report["global_basis_count"] - 1,
            )
        self.assertEqual(report["numeric_tolerance"], 1.0e-10)
        self.assertFalse(report["global_independence_certified"])

    def test_source_term_order_invariance_and_raw_neutron_phase_matters(self):
        request = make_sigma_lambda_example()
        shuffled = replace(
            request,
            source=SourceLinearCombination(OperatorLinearCombination(
                tuple(reversed(request.source.combination.terms)),
            )),
        )
        first = reduce_linear_combination(request)
        second = reduce_linear_combination(shuffled)
        for sink in first["sink_names"]:
            self.assertEqual(_basis_coefficients(first, sink), _basis_coefficients(second, sink))

        wrong_phase_terms = list(request.source.combination.terms)
        wrong_phase_terms[-1] = replace(
            wrong_phase_terms[-1], coefficient=-wrong_phase_terms[-1].coefficient,
        )
        wrong_phase = replace(
            request,
            source=SourceLinearCombination(OperatorLinearCombination(tuple(wrong_phase_terms))),
        )
        wrong_report = reduce_linear_combination(wrong_phase)
        for sink in wrong_report["sink_names"]:
            self.assertTrue(any(
                basis["rows"][sink]["exact_zero"] is False
                for basis in wrong_report["global_basis"]
            ))

    def test_lambdap_example_uses_the_shared_old_kernel(self):
        request = make_sigma_lambda_example()
        lambdap = replace(request, sinks=(request.sinks[0],))
        report = reduce_linear_combination(lambdap)
        self.assertEqual(report["sink_names"], ["Lambda1p"])
        self.assertEqual(report["wick_count"], 36)
        self.assertEqual(len(report["global_batches"]), 1)
        self.assertTrue(report["exactness"]["all_rows_exact_zero"])


if __name__ == "__main__":
    unittest.main()
