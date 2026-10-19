"""Input, transporter, and numerical regressions for construction modes."""

import ast
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kernels import correlator_wick_normalizer as core
from verify_code import check_basis_vs_original
from verify_code.generic_check_numeric import RandomTensorBank, audit_one
from kernels.wick_reducer import expression_output as output
from kernels.wick_reducer import operator_model as model, peram_incidence, reducer


def _normal_example(name, *, labeled=False):
    operators = model.operator_library()[name]
    if not labeled:
        return tuple(replace(operator, mode="normal") for operator in operators)
    return tuple(replace(
        operator, mode="normal", external_tag=f"p_{index}",
        contraction_metadata=core.normalize_contraction_metadata(
            momentum=[index + 1, 0, -1],
        ),
    ) for index, operator in enumerate(operators))


def _first_key(operators):
    _fields, quarks, antiquarks = model.fields_for_operators(operators)
    term = next(core.enumerate_wicks(quarks, antiquarks))
    classified_quarks = tuple(
        replace(field, flavor=model.flavor_class(field.flavor, True))
        for field in quarks
    )
    classified_antiquarks = tuple(
        replace(field, flavor=model.flavor_class(field.flavor, True))
        for field in antiquarks
    )
    return reducer.canonical_factor_graph_key(
        operators, quarks, antiquarks, term,
        classified_quarks, classified_antiquarks,
    )[0]


class OperatorModeInputTests(unittest.TestCase):
    def test_default_matches_explicit_distillation_counts_and_output(self):
        for name, counts in (("pn", (36, 18, 18)), ("delta", (6, 2, 4)), ("pip", (24, 24, 0))):
            with self.subTest(name=name):
                default = model.operator_library()[name]
                explicit = tuple(replace(operator, mode="distillation") for operator in default)
                self.assertTrue(all(operator.mode == "distillation" for operator in default))
                left = reducer.reduce_operators(default, label=name, mode="detailed")
                right = reducer.reduce_operators(explicit, label=name, mode="detailed")
                self.assertEqual(tuple(left[key] for key in (
                    "wick_count", "independent_count", "dependent_count",
                )), counts)
                for key in ("independent_wicks", "dependent_relations", "graph_records"):
                    self.assertEqual(left[key], right[key])
                self.assertEqual(
                    output.render_python_output(left, default, name=name, mode="check"),
                    output.render_python_output(right, explicit, name=name, mode="check"),
                )

    def test_both_field_builders_propagate_all_endpoint_labels(self):
        operators = tuple(model.make_operator(
            side, kind, "custom", f"{side}_{kind}",
            ("u", "u", "u") if kind == "baryon" else ("u",),
            "Cgamma5" if kind == "baryon" else "gamma5",
            mode="normal", external_tag=f"{side}_{kind}_p", momentum=[1, 2, 3],
            projector="kept", phase=1,
        ) for side in ("sink", "source") for kind in ("baryon", "meson"))
        for build_fields in (model.fields_for_operators, core._fields):
            with self.subTest(builder=build_fields.__name__):
                fields, _quarks, antiquarks = build_fields(operators)
                self.assertEqual(len(fields), 10)
                self.assertEqual(len(antiquarks), 5)
                for field in fields:
                    operator = operators[field.operator]
                    self.assertEqual(field.mode, "normal")
                    self.assertEqual(field.endpoint_descriptor, (
                        "x", operator.external_tag, (("momentum", (1, 2, 3)),),
                    ))
                    self.assertEqual(core.field_endpoint_metadata(field)["momentum"], [1, 2, 3])
                    hash(field.endpoint_descriptor)
        for operator in operators:
            self.assertEqual(core.operator_factor_tag(operator), "")
            self.assertEqual(dict(core.operator_factor_metadata(operator)), {"phase": 1, "projector": "kept"})
            self.assertIn("momentum", dict(operator.contraction_metadata))

    def test_json_report_round_trip_preserves_mode_and_endpoint_metadata(self):
        for fixture in ("delta", "pip"):
            with self.subTest(fixture=fixture):
                operators = _normal_example(fixture, labeled=True)
                serialized = core.process(operators, max_wicks=1)
                self.assertEqual(serialized["operator_mode"], "normal")
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "operators.json"
                    path.write_text(json.dumps({"name": "normal_input", "operators": serialized["operators"]}), encoding="utf-8")
                    name, recovered, _matrices = core._from_json(path)
                self.assertEqual(name, "normal_input")
                self.assertEqual(recovered, operators)
                report = reducer.reduce_operators(recovered, mode="detailed")
                self.assertEqual(report["operator_mode"], "normal")
                json.dumps(report)
                for graph in report["graph_records"]:
                    for line in graph["peram_lines"]:
                        self.assertIn("_x_x", line["peram_class"])
                        for endpoint in (line["q"], line["bar"]):
                            self.assertEqual(endpoint["mode"], "normal")
                            self.assertEqual(endpoint["representation"], "x")
                            self.assertEqual(endpoint["momentum"], [endpoint["operator"] + 1, 0, -1])

    def test_invalid_modes_and_mixed_correlators_are_rejected(self):
        for constructor in (model.make_operator, core.op):
            with self.subTest(constructor=constructor.__name__):
                with self.assertRaisesRegex(ValueError, "distillation.*normal|normal.*distillation"):
                    constructor("sink", "meson", "pi", "pi", ("u",), "gamma5", mode="invalid")
        valid = model.operator_library()["delta"]
        with self.assertRaises(ValueError):
            replace(valid[0], mode="invalid")
        with self.assertRaises(ValueError):
            core.Operator("sink", "meson", "pi", "pi", ("u",), "gamma5", (0, 1), None, "", mode="invalid")
        mixed = (replace(valid[0], mode="normal"), valid[1])
        checks = (
            lambda: model.fields_for_operators(mixed),
            lambda: core._fields(mixed),
            lambda: core.process(mixed),
            lambda: reducer.reduce_operators(mixed, mode="coarse"),
            lambda: reducer.reduce_operators(mixed, mode="detailed"),
            lambda: output.normalize_correlator(output.split_operators(mixed)),
            lambda: output.contraction_expression(mixed, (0, 1, 2)),
        )
        for check in checks:
            with self.subTest(check=check):
                with self.assertRaisesRegex(ValueError, "same operator mode|mixed"):
                    check()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "same operator mode|mixed"):
                output.write_output(output.split_operators(mixed), Path(directory) / "mixed.py")
            path = Path(directory) / "mixed.json"
            path.write_text(json.dumps({"operators": [
                {"side": operator.side, "kind": "baryon", "species": "Delta", "label": operator.label,
                 "flavors": list(operator.flavors), "gamma": operator.gamma, "mode": operator.mode}
                for operator in mixed
            ]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same operator mode|mixed"):
                core._from_json(path)


class OperatorModeClassificationTests(unittest.TestCase):
    def test_momentum_distinguishes_profiles_graphs_edges_and_symbols(self):
        first = _normal_example("delta")
        second = (replace(first[0], contraction_metadata=core.normalize_contraction_metadata(momentum=1)), first[1])
        _fields, quarks, antiquarks = model.fields_for_operators(first)
        _other_fields, other_quarks, other_antiquarks = model.fields_for_operators(second)
        term = next(core.enumerate_wicks(quarks, antiquarks))
        self.assertEqual(
            [item.pairing for item in core.enumerate_wicks(quarks, antiquarks)],
            [item.pairing for item in core.enumerate_wicks(other_quarks, other_antiquarks)],
        )
        self.assertNotEqual(model.profile(term, quarks, antiquarks, True)[0], model.profile(term, other_quarks, other_antiquarks, True)[0])
        self.assertNotEqual(_first_key(first), _first_key(second))
        self.assertNotEqual(peram_incidence._edge_color(quarks[0], antiquarks[0], True), peram_incidence._edge_color(other_quarks[0], other_antiquarks[0], True))
        self.assertNotEqual(output._peram_symbol(quarks[0], antiquarks[0]), output._peram_symbol(other_quarks[0], other_antiquarks[0]))
        self.assertEqual(reducer._operator_color(first[0], True), reducer._operator_color(second[0], True))

    def test_final_line_matching_requires_matching_endpoint_momentum(self):
        operators = _normal_example("delta")
        _fields, quarks, antiquarks = model.fields_for_operators(operators)
        term = next(core.enumerate_wicks(quarks, antiquarks))
        changed = [replace(quarks[0], endpoint_descriptor=("x", "", (("momentum", (1, 0, 0)),))), *quarks[1:]]
        candidates = reducer._line_candidates(
            operators, quarks, antiquarks, term, term,
            changed, antiquarks, (0, 1), True,
        )
        self.assertEqual(candidates[0], ())
        self.assertEqual(list(reducer._iter_line_maps(candidates, 32)), [])

    def test_hash_uses_ordered_labels_and_canonical_momentum(self):
        def meson(label, momentum):
            return model.make_operator("sink", "meson", "pi", label, ("u",), "gamma5", mode="normal", momentum=momentum)
        _fields, quarks, antiquarks = model.fields_for_operators((meson("a", [1, 0, 0]), meson("b", (1, 0, 0))))
        self.assertEqual(output._peram_symbol(quarks[0], antiquarks[1]), output._peram_symbol(quarks[1], antiquarks[0]))
        _fields, quarks, antiquarks = model.fields_for_operators((meson("a", 1), meson("b", "1")))
        self.assertNotEqual(output._peram_symbol(quarks[0], antiquarks[1]), output._peram_symbol(quarks[1], antiquarks[0]))
        _fields, quarks, antiquarks = model.fields_for_operators((meson("a", None), meson("b", 0)))
        self.assertEqual(output._peram_symbol(quarks[0], antiquarks[0]), "peram_u_sin_sin_x_x")
        self.assertIn("_p_", output._peram_symbol(quarks[1], antiquarks[1]))

    def test_all_directions_and_isospin_flavor_names(self):
        operators = tuple(model.make_operator(
            side, "meson", "pi", side, ("d",), "gamma5", mode="normal",
        ) for side in ("sink", "source"))
        _fields, quarks, antiquarks = model.fields_for_operators(operators)
        for quark in quarks:
            for anti in antiquarks:
                direction = core.peram_direction(quark, anti)
                self.assertEqual(output._peram_symbol(quark, anti), f"peram_u_{direction}_x_x")
                self.assertEqual(output._peram_symbol(quark, anti, isospin_degenerate=False), f"peram_d_{direction}_x_x")
        self.assertEqual({core.peram_direction(q, anti) for q in quarks for anti in antiquarks}, set(model.PERAM_ORDER))


class OperatorModeNumericTests(unittest.TestCase):
    def test_normal_tensor_bank_uses_actual_epsilon_delta_and_three_colors(self):
        bank = RandomTensorBank(17, 6)
        epsilon = bank.get("epsilon_color")
        self.assertEqual(epsilon.shape, (3, 3, 3))
        self.assertEqual(np.count_nonzero(epsilon), 6)
        self.assertEqual(epsilon[0, 1, 2], 1)
        self.assertEqual(epsilon[1, 0, 2], -1)
        np.testing.assert_array_equal(bank.get("delta_color"), np.eye(3))
        self.assertEqual(bank.get("peram_u_sin_sou_x_x_p_label").shape, (3, 4, 3, 4))
        self.assertEqual(bank.get("peram_u_sin_sou").shape, (6, 4, 6, 4))

    def test_normal_direct_wicks_match_basis_with_and_without_isospin(self):
        for name, labeled in (("delta", False), ("pipi", False), ("pip", True), ("pn", True)):
            for isospin in (True, False):
                with self.subTest(name=name, labeled=labeled, isospin=isospin):
                    result = audit_one(
                        name, _normal_example(name, labeled=labeled),
                        seed=20261013, color_dimension=6,
                        isospin_degenerate=isospin,
                    )
                    self.assertEqual(result["operator_mode"], "normal")
                    self.assertEqual(result["isospin_symmetric"], isospin)
                    self.assertEqual(result["color_dimension"], 3)
                    self.assertIsNone(result["eigenvector_dimension"])
                    self.assertLess(result["max_relative_residual"], 1.0e-10)

    def test_normal_check_writer_matches_streamed_output_and_executes(self):
        operators = _normal_example("pip", labeled=True)
        report = reducer.reduce_operators(
            operators, label="normal_check", mode="detailed",
            options=reducer.ReductionOptions(isospin_degenerate=False),
        )
        expected = output.render_python_output(report, operators, name="normal_check", mode="check")
        self.assertIn("epsilon_color", expected)
        self.assertIn("delta_color", expected)
        self.assertNotIn("VVV_sink", expected)
        self.assertNotIn("conj_VDV_source", expected)
        self.assertIn("_x_x_p_", expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal.py"
            result = output.write_output(
                output.split_operators(operators), path, name="normal_check",
                mode="check", isospin_degenerate=False,
            )
            self.assertEqual(path.read_text(encoding="utf-8"), expected)
            self.assertTrue(result["report"]["check_output_streamed"])
        ast.parse(expected)
        values, _projections = check_basis_vs_original._random_values(
            operators, 6, 97, isospin_degenerate=False,
        )
        namespace = dict(values)
        exec(compile(expected, "normal_check.py", "exec"), namespace)
        self.assertEqual(len(namespace["normal_check_origen_wick_raw"]), report["wick_count"])
        for wick_index, raw in namespace["normal_check_origen_wick_raw"].items():
            np.testing.assert_allclose(
                namespace["normal_check_origen_wick_sign"][wick_index] * raw,
                namespace["normal_check_origen_wick"][wick_index],
                rtol=1.0e-10, atol=1.0e-10,
            )
        numeric = check_basis_vs_original._check_one_dimension(operators, report, size=6, seed=97)
        self.assertLess(numeric["max_relative_residual"], 1.0e-10)
        self.assertTrue(numeric["passed"])

    def test_audit_check_file_uses_report_isospin_policy(self):
        operators = _normal_example("pip", labeled=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal_audit.py"
            result = audit_one(
                "normal_audit", operators, seed=47, color_dimension=6,
                write_check=path, isospin_degenerate=False,
            )
            content = path.read_text(encoding="utf-8")
        self.assertEqual(result["color_dimension"], 3)
        self.assertFalse(result["isospin_symmetric"])
        self.assertIn("peram_d_", content)
        self.assertIn("peram_u_", content)
        ast.parse(content)


if __name__ == "__main__":
    unittest.main()
