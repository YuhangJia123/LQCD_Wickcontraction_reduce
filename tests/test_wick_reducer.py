import unittest
import tempfile
import sys
from dataclasses import replace
from pathlib import Path
import re

QCT_DIR = str(Path(__file__).resolve().parents[1])
if QCT_DIR not in sys.path:
    sys.path.insert(0, QCT_DIR)

from kernels import correlator_wick_normalizer as core
from kernels.wick_reducer import expression_output as output
from kernels.wick_reducer import reducer
from kernels.wick_reducer import operator_model as model


class WickReducerTests(unittest.TestCase):
    def test_compact_bucket_reads_pairing_and_cached_operator_map_in_place(self):
        bucket = reducer.CompactWickBucket(1, 3, 1, 2)
        term = core.WickTerm((2, 0, 1), -1)
        bucket.append(7, term, (1, 0))

        record = bucket[0]
        self.assertEqual(record.wick_index, 7)
        self.assertEqual(reducer._term_from_compact(record), term)
        self.assertEqual(record.operator_map(), (1, 0))

    def test_bipartite_refinement_is_permutation_invariant_and_has_exact_fallback(self):
        cells = [
            [[0], [0], [1]],
            [[0], [1], [2]],
            [[2], [2], [2]],
        ]
        row_order, column_order = reducer._refined_bipartite_orders(cells)
        canonical = tuple(
            tuple(tuple(cells[row][column]) for column in column_order)
            for row in row_order
        )

        row_permutation = (2, 0, 1)
        column_permutation = (1, 2, 0)
        permuted = [
            [cells[row][column] for column in column_permutation]
            for row in row_permutation
        ]
        other_rows, other_columns = reducer._refined_bipartite_orders(permuted)
        other_canonical = tuple(
            tuple(tuple(permuted[row][column]) for column in other_columns)
            for row in other_rows
        )
        self.assertEqual(canonical, other_canonical)
        self.assertIsNone(reducer._refined_bipartite_orders([[[0], [0]], [[0], [0]]]))

    def test_requested_wick_counts(self):
        expected = {
            "pipi": 24,
            "pn": 36,
            "ppn": 2880,
            "delta": 6,
            "delta++delta-": 36,
            "pip": 24,
            "pip-p": 6,
            "pn_omega_pi+": 3456,
        }
        for name, count in expected.items():
            with self.subTest(name=name):
                report = reducer.reduce_case(name, mode="coarse")
                self.assertEqual(report["wick_count"], count)

    def test_delta_enumerates_multiple_peram_transports(self):
        report = reducer.reduce_case("delta", mode="detailed")
        self.assertEqual(report["wick_count"], 6)
        self.assertGreaterEqual(report["multi_solution"]["max_transporter_solutions"], 2)
        self.assertGreater(report["multi_solution"]["sink_rank_checks"], 0)
        self.assertEqual(
            report["independent_count"] + report["dependent_count"],
            report["wick_count"],
        )
        self.assertEqual(
            report["independent_wicks"][0]["coefficient"],
            [2.0, 0.0],
        )
        first_relation = report["dependent_relations"][0]
        self.assertEqual(first_relation["transporter_tie_count"], 2)
        self.assertEqual(first_relation["sink_orbit"]["sink_solution_rank"], 1)

    def test_zero_sink_overlap_short_circuits_rank(self):
        matrices = core._euclidean_gamma_matrices()
        axes = ((0, "q"), (1, "q"))
        reference = {
            "axes": axes,
            "factors": ((axes[0], axes[1], matrices["gamma5"]),),
            "signature": (("gamma5", axes[0], axes[1]),),
        }
        orthogonal = {
            "axes": axes,
            "factors": ((axes[0], axes[1], matrices["gamma1"]),),
            "signature": (("gamma1", axes[0], axes[1]),),
        }
        state = reducer.TransportState((), (), (), orthogonal, {}, 1, 1, 1, {})
        result = reducer._sink_orbit_diagnostics(reference, [state], reducer.TOLERANCE)
        self.assertTrue(result["zero_overlap_short_circuit"])
        self.assertFalse(result["sink_rank_checked"])
        self.assertIsNone(result["sink_solution_rank"])

        combined_matrix = tuple(
            tuple(left + right for left, right in zip(row_a, row_b))
            for row_a, row_b in zip(matrices["gamma5"], matrices["gamma1"])
        )
        combined = {
            "axes": axes,
            "factors": ((axes[0], axes[1], combined_matrix),),
            "signature": (("gamma5+gamma1", axes[0], axes[1]),),
        }
        partial = reducer.TransportState((), (), (), combined, {}, 1, 1, 1, {})
        rank_report = reducer._sink_orbit_diagnostics(
            reference, [state, partial], reducer.TOLERANCE
        )
        self.assertEqual(rank_report["sink_solution_rank"], 2)
        self.assertFalse(rank_report["sink_baseline_tests"][1]["proportional_to_baseline"])
        self.assertGreater(
            abs(rank_report["sink_baseline_tests"][1]["normalized_overlap"][0]),
            0.1,
        )

    def test_bare_names_are_renamed_and_transpose_is_reported(self):
        operators = model.operator_library()["ppn"]
        fields, quarks, antiquarks = core._fields(operators)
        term = next(core.enumerate_wicks(quarks, antiquarks))
        operator_maps = tuple(
            {index: target for index, target in enumerate(mapping)}
            for mapping in reducer._operator_maps(operators, True)
        )
        original_key = reducer.peram_incidence.incidence_key(
            operators, quarks, antiquarks, term, operator_maps=operator_maps
        )
        renamed_quarks = [
            replace(field, axis=f"q{index}") if field.bare else field
            for index, field in enumerate(quarks)
        ]
        renamed_bars = [
            replace(field, axis=f"b{index}") if field.bare else field
            for index, field in enumerate(antiquarks)
        ]
        renamed_key = reducer.peram_incidence.incidence_key(
            operators, renamed_quarks, renamed_bars, term,
            operator_maps=operator_maps,
        )
        self.assertEqual(original_key, renamed_key)

        maps, truncated = reducer._iter_transport_maps(
            operators, quarks, antiquarks, term, term, isospin_degenerate=True
        )
        self.assertFalse(truncated)
        witnesses = [
            reducer._bare_axis_map(quarks, antiquarks, term, term, line_map)
            for _operator_map, line_map in maps
        ]
        self.assertTrue(any(any(source != target for source, target in item.items()) for item in witnesses))

    def test_isospin_classes_merge_only_u_and_d(self):
        self.assertEqual(reducer.model.flavor_class("u", True), "ud")
        self.assertEqual(reducer.model.flavor_class("d", True), "ud")
        self.assertEqual(reducer.model.flavor_class("s", True), "s")
        self.assertEqual(reducer.model.flavor_class("u", False), "u")
        self.assertEqual(reducer.model.flavor_class("d", False), "d")

    def test_generic_factor_key_omits_species_labels(self):
        operators = model.operator_library()["pn"]
        _fields, quarks, antiquarks = model.fields_for_operators(operators)
        term = next(core.enumerate_wicks(quarks, antiquarks))
        q_classes = tuple(model.flavor_class(field.flavor, True) for field in quarks)
        anti_classes = tuple(model.flavor_class(field.flavor, True) for field in antiquarks)
        classified_quarks = tuple(
            replace(field, flavor=flavor)
            for field, flavor in zip(quarks, q_classes)
        )
        classified_antiquarks = tuple(
            replace(field, flavor=flavor)
            for field, flavor in zip(antiquarks, anti_classes)
        )
        isospin_key, witness = reducer.canonical_factor_graph_key(
            operators, quarks, antiquarks, term,
            classified_quarks, classified_antiquarks,
            isospin_degenerate=True,
        )
        exact_quarks = tuple(
            replace(field, flavor=field.flavor)
            for field in quarks
        )
        exact_antiquarks = tuple(
            replace(field, flavor=field.flavor)
            for field in antiquarks
        )
        exact_key, _ = reducer.canonical_factor_graph_key(
            operators, quarks, antiquarks, term,
            exact_quarks, exact_antiquarks,
            isospin_degenerate=False,
        )
        self.assertEqual(len(witness), len(operators))
        self.assertNotEqual(isospin_key, exact_key)
        self.assertEqual(isospin_key[0][0][2], "")
        self.assertEqual(exact_key[0][0][2], "")

    def test_generic_vertex_color_omits_species_labels(self):
        proton = model.make_operator(
            "sink", "baryon", "p", "sink_p", ("u", "u", "d"),
            "Cgamma5", (1, 2), open_slot=0,
        )
        neutron = replace(proton, species="n", label="sink_n")
        delta = replace(proton, species="delta", label="sink_delta")
        pion = model.make_operator(
            "sink", "meson", "pi+", "sink_pi", ("d", "u"),
            "gamma5", (0, 1),
        )
        kaon = replace(pion, species="K+", label="sink_K")

        p_color = reducer._operator_color(proton, True)
        n_color = reducer._operator_color(neutron, True)
        self.assertEqual(p_color, n_color)
        delta_color = reducer._operator_color(delta, True)
        self.assertEqual(p_color, delta_color)
        self.assertNotEqual(p_color, reducer._operator_color(pion, True))
        self.assertEqual(
            reducer._operator_color(pion, True),
            reducer._operator_color(kaon, True),
        )

        incidence = reducer.peram_incidence
        self.assertEqual(
            incidence._operator_color(proton),
            incidence._operator_color(neutron),
        )
        delta_incidence_color = incidence._operator_color(delta)
        pion_color = incidence._operator_color(pion)
        kaon_color = incidence._operator_color(kaon)
        self.assertIn("species:|", delta_incidence_color)
        self.assertIn("species:|", pion_color)
        self.assertIn("species:|", kaon_color)
        self.assertEqual(pion_color, kaon_color)

    def test_charged_meson_fields_keep_distinct_antiquark_and_quark_flavors(self):
        operators = (
            reducer.model.make_operator(
                "sink", "meson", "pi+", "sink_pi+", ("d", "u"),
                "gamma5", (0, 1),
            ),
            reducer.model.make_operator(
                "source", "meson", "pi+", "source_pi+", ("u", "d"),
                "gamma5", (0, 1),
            ),
        )
        fields, quarks, antiquarks = reducer.model.fields_for_operators(operators)
        self.assertEqual(
            [(field.kind, field.flavor) for field in fields],
            [("bar", "d"), ("q", "u"), ("bar", "u"), ("q", "d")],
        )
        self.assertEqual(len(list(core.enumerate_wicks(quarks, antiquarks))), 1)

    def test_strange_peram_lines_do_not_match_light_lines_in_isospin_mode(self):
        operators = (
            reducer.model.make_operator(
                "sink", "baryon", "B", "sink_B", ("u", "d", "s"),
                "Cgamma5", (0, 1), open_slot=2,
            ),
            reducer.model.make_operator(
                "source", "baryon", "B", "source_B", ("u", "d", "s"),
                "Cgamma5_dagger", (0, 1), open_slot=2,
            ),
        )
        _fields, quarks, antiquarks = reducer.model.fields_for_operators(operators)
        term = next(core.enumerate_wicks(quarks, antiquarks))
        candidates = reducer._line_candidates(
            operators, quarks, antiquarks, term, term, quarks, antiquarks,
            (0, 1), True,
        )
        strange_line = next(index for index, field in enumerate(quarks) if field.flavor == "s")
        down_line = next(index for index, field in enumerate(quarks) if field.flavor == "d")
        self.assertEqual(candidates[strange_line], (strange_line,))
        self.assertEqual(candidates[down_line], (0, down_line))

        exact_candidates = reducer._line_candidates(
            operators, quarks, antiquarks, term, term, quarks, antiquarks,
            (0, 1), False,
        )
        self.assertEqual(exact_candidates[down_line], (down_line,))

    def test_concrete_custom_four_by_four_matrix_is_used_directly(self):
        matrix = (
            (1, 0, 0, 0),
            (0, 2, 0, 0),
            (0, 0, 3, 0),
            (0, 0, 0, 4),
        )
        operators = (
            reducer.model.make_operator(
                "sink", "meson", "M", "sink_M", ("u", "u"),
                "g1g5", (0, 1),
            ),
            reducer.model.make_operator(
                "source", "meson", "M", "source_M", ("u", "u"),
                "g1g5", (0, 1),
            ),
        )
        report = reducer.reduce_case(
            "custom_matrix", operators, mode="detailed",
            gamma_matrices={"g1g5": matrix},
        )
        self.assertEqual(report["wick_count"], 2)
        self.assertEqual(
            report["independent_count"] + report["dependent_count"], 2
        )

        with self.assertRaisesRegex(ValueError, "4x4"):
            reducer.reduce_case(
                "custom_matrix", operators, mode="independent",
                gamma_matrices={"g1g5": ((1, 0), (0, 1))},
            )

    def test_source_payload_is_built_only_after_sink_stage(self):
        operators = reducer.model.operator_library()["delta"]
        fields, quarks, antiquarks = reducer.model.fields_for_operators(operators)
        term = next(core.enumerate_wicks(quarks, antiquarks))
        state = reducer._transport_state(
            operators, fields, quarks, antiquarks, term, term,
            (0, 1), tuple(range(len(quarks))), core._euclidean_gamma_matrices(),
        )
        self.assertIsNotNone(state.sink)
        self.assertIsNone(state.source)

        source = reducer._ensure_source_payload(
            state, operators, fields, quarks, antiquarks, term,
            core._euclidean_gamma_matrices(),
        )
        self.assertIs(source, state.source)

    def test_generic_output_pair_and_full_ppn_indices(self):
        operators = model.operator_library()["ppn"]
        self.assertEqual(operators[3].flavors, ("u", "d", "u"))
        self.assertEqual(operators[5].flavors, ("d", "u", "d"))
        correlator = output.split_operators(operators)
        self.assertEqual(len(output.normalize_correlator(correlator)), 6)
        fields, quarks, antiquarks = model.fields_for_operators(operators)
        first = next(core.enumerate_wicks(quarks, antiquarks))
        equation, operands, free = output.contraction_expression(operators, first.pairing)
        self.assertEqual(free, tuple("ABCDEF"))
        self.assertTrue(equation.endswith("->ABCDEF"))
        self.assertIn("C_g5", operands)
        self.assertIn("C_g5_dagger", operands)
        self.assertIn("peram_u_sin_sou", operands)
        self.assertIn("VVV_sink", operands)
        self.assertIn("conj_VVV_source", operands)
        self.assertNotIn("VVV_sink_p", operands)

    def test_elemental_names_use_momentum_tag_not_species(self):
        operators = model.operator_library()["ppn"]
        sink_names = [output._elemental_symbol(operator) for operator in operators[:3]]
        source_names = [output._elemental_symbol(operator) for operator in operators[3:]]
        self.assertEqual(set(sink_names), {"VVV_sink"})
        self.assertEqual(set(source_names), {"conj_VVV_source"})
        tagged = replace(operators[0], external_tag="1")
        self.assertEqual(output._elemental_symbol(tagged), "VVV_sink_1")
        self.assertNotIn("p", output._elemental_symbol(tagged).split("_"))
        momentum_tagged = replace(
            operators[0],
            contraction_metadata=(("momentum", (1, 0, 0)), ("projector", "P+")),
        )
        self.assertNotEqual(
            output._elemental_symbol(operators[0]),
            output._elemental_symbol(momentum_tagged),
        )

    def test_scalar_phase_and_clebsch_are_applied_to_contract_calls(self):
        operators = list(model.operator_library()["pip"])
        operators[0] = replace(
            operators[0], contraction_metadata=(("phase", ("complex", 0.0, 1.0)),)
        )
        report = reducer.reduce_operators(
            tuple(operators), label="weighted", mode="independent",
        )
        calls, _axes = output._render_contracts(
            report["independent_wicks"], operators,
            isospin_degenerate=True,
        )
        self.assertIn("complex(0, 1) * contract(", calls[0])

    def test_contract_calls_are_one_line(self):
        call = output._contract_call("ab,bc->ac", ("gamma", "peram"))
        self.assertEqual(call, "    contract('ab,bc->ac', gamma, peram),")
        self.assertNotIn("\n", call)

    def test_source_slots_are_taken_verbatim(self):
        source = model.make_operator(
            "source", "baryon", "custom", "source_custom", ("s", "u", "d"),
            "g1g5", (1, 2), open_slot=0,
        )
        self.assertEqual(source.flavors, ("s", "u", "d"))
        fields, _quarks, antiquarks = model.fields_for_operators((source,))
        self.assertEqual([field.flavor for field in antiquarks], ["s", "u", "d"])

    def test_check_output_has_raw_graphs_and_transpose_relations(self):
        operators = model.operator_library()["delta"]
        report = reducer.reduce_case("delta", operators, mode="detailed")
        shuffled_report = dict(report)
        shuffled_report["graph_records"] = list(reversed(report["graph_records"]))
        text = output.render_python_output(
            shuffled_report, operators, name="delta", mode="check",
        )
        self.assertLess(text.index("delta_coefficients = ["), text.index("delta_origen_wick_raw[0]"))
        self.assertIn("delta_origen_wick_raw[0] = contract(", text)
        self.assertIn("delta_origen_wick_sign[0] =", text)
        self.assertIn("delta_origen_wick[0] = (1 * delta_basis[0])", text)
        self.assertNotIn("delta_origen_wick_idx", text)
        self.assertIn("delta_origen_wick_to_independent", text)
        self.assertIn("delta_basis_contract = {}", text)
        self.assertIn("delta_basis = {}", text)
        self.assertIn("delta_result = sum(", text)
        self.assertIn("delta_basis[1] = (-delta_basis_contract[3])", text)
        self.assertNotIn("delta_basis_wick_sign", text)
        self.assertIn("'basis_wick_id':", text)
        self.assertIn("'basis_index':", text)
        self.assertEqual(len(report["graph_records"]), report["wick_count"])
        raw_order = [int(value) for value in re.findall(r"delta_origen_wick_raw\[(\d+)\] = contract", text)]
        self.assertEqual(raw_order, list(range(report["wick_count"])))
        coefficient_line = next(line for line in text.splitlines() if line.startswith("delta_coefficients = "))
        self.assertNotIn("\n", coefficient_line)
        self.assertNotIn("delta_raw_wick_index", text)
        for relation in report["dependent_relations"]:
            self.assertEqual(
                len(relation["basis_transports"]),
                len(relation["basis_wick_indices"]),
            )
        with tempfile.TemporaryDirectory() as directory:
            result = output.write_output(
                output.split_operators(operators), Path(directory) / "delta_check.py",
                name="delta", mode="check",
            )
        self.assertEqual(result["reduced_term_count"], 2)

    def test_independent_python_export_groups_bare_axis_transposes(self):
        operators = model.operator_library()["ppn"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ppn.py"
            result = output.write_output(
                output.split_operators(operators), path,
                name="ppn", mode="independent",
            )
            text = path.read_text(encoding="utf-8")
        self.assertEqual(result["wick_count"], 2880)
        self.assertEqual(result["independent_count"], 219)
        self.assertGreater(result["reduced_term_count"], result["independent_count"])
        self.assertIn("ppn_basis_contract = {}", text)
        self.assertIn("ppn_basis = {}", text)
        self.assertIn("ppn_coefficients = [", text)
        self.assertIn("ppn_result = sum(", text)
        self.assertNotIn("ppn_idx", text)
        self.assertNotIn("ppn_basis_wick_sign", text)
        coefficient_line = next(line for line in text.splitlines() if line.startswith("ppn_coefficients = "))
        self.assertNotIn("\n", coefficient_line)
        self.assertIn(".transpose(", text)
        self.assertIn("peram_u_sin_sou", text)
        self.assertIn("VVV_sink: shared default momentum", text)


if __name__ == "__main__":
    unittest.main()
