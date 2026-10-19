import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kernels import correlator_wick_normalizer as core
from kernels.wick_reducer import ReductionOptions, reduce_operators
from kernels.wick_reducer import operator_model as model
from kernels.wick_reducer import reducer, peram_incidence


class GenericReductionApiTests(unittest.TestCase):
    def test_custom_operator_entry_does_not_require_system_name(self):
        operators = model.operator_library()["pn"]
        report = reduce_operators(
            operators,
            label="arbitrary_input",
            mode="independent",
            options=ReductionOptions(),
        )
        self.assertEqual(report["name"], "arbitrary_input")
        self.assertEqual(report["wick_count"], 36)
        self.assertEqual(
            report["independent_count"] + report["dependent_count"],
            report["wick_count"],
        )
        self.assertTrue(report["factor_graph"])

    def test_isospin_policy_is_independent_of_operator_label(self):
        operators = model.operator_library()["pn"]
        symmetric = reduce_operators(
            operators,
            label="input_a",
            mode="coarse",
            options=ReductionOptions(isospin_degenerate=True),
        )
        exact = reduce_operators(
            operators,
            label="input_b",
            mode="coarse",
            options=ReductionOptions(isospin_degenerate=False),
        )
        self.assertEqual(symmetric["wick_count"], exact["wick_count"])
        self.assertTrue(symmetric["isospin_symmetric"])
        self.assertFalse(exact["isospin_symmetric"])

    def test_no_options_uses_decoupled_generic_policy(self):
        report = reduce_operators(
            model.operator_library()["pn"],
            label="no_policy_name",
            mode="coarse",
        )
        self.assertTrue(report["factor_graph"])

    def test_generic_key_omits_species_but_keeps_flavor_edges(self):
        sink = model.make_operator(
            "sink", "meson", "pi", "sink_pi", ("u", "u"), "gamma5",
        )
        source = model.make_operator(
            "source", "meson", "K", "source_K", ("u", "u"), "gamma5",
        )
        operators = (sink, source)
        fields, quarks, antiquarks = model.fields_for_operators(operators)
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
        generic_key, _ = reducer.canonical_factor_graph_key(
            operators, quarks, antiquarks, term,
            classified_quarks, classified_antiquarks,
            isospin_degenerate=True,
        )
        self.assertEqual(generic_key[0][0][2], "")
        self.assertEqual(generic_key[0][1][2], "")
        self.assertIn("species:|", peram_incidence._operator_color(sink))
        self.assertIn("species:|", peram_incidence._operator_color(source))

        def first_key(pair):
            pair_fields, pair_quarks, pair_antiquarks = model.fields_for_operators(pair)
            del pair_fields
            pair_term = next(core.enumerate_wicks(pair_quarks, pair_antiquarks))
            pair_q = tuple(
                replace(field, flavor=model.flavor_class(field.flavor, True))
                for field in pair_quarks
            )
            pair_anti = tuple(
                replace(field, flavor=model.flavor_class(field.flavor, True))
                for field in pair_antiquarks
            )
            return reducer.canonical_factor_graph_key(
                pair, pair_quarks, pair_antiquarks, pair_term,
                pair_q, pair_anti, isospin_degenerate=True,
            )[0]

        pion = (sink, source)
        kaon = (
            model.make_operator(
                "sink", "meson", "K+", "sink_K", ("s", "u"), "gamma5",
            ),
            model.make_operator(
                "source", "meson", "K-", "source_K", ("u", "s"), "gamma5",
            ),
        )
        # A label-only rename has the same key; changing u to s changes the
        # directed peram edge colors and therefore the canonical key.
        self.assertEqual(first_key(pion), first_key((
            replace(sink, species="K_label"),
            replace(source, species="pi_label"),
        )))
        self.assertNotEqual(first_key(pion), first_key(kaon))

    def test_old_species_switch_is_not_part_of_the_generic_api(self):
        with self.assertRaises(TypeError):
            legacy_keyword = {"ignore_" + "hadron_species": True}
            reduce_operators(
                model.operator_library()["pn"],
                **legacy_keyword,
            )

    def test_other_flavors_are_not_collapsed_by_isospin_policy(self):
        sink = model.make_operator(
            "sink", "meson", "S", "sink_S", ("s", "s"), "gamma5",
        )
        source = model.make_operator(
            "source", "meson", "U", "source_U", ("u", "u"), "gamma5",
        )
        _fields, quarks, antiquarks = model.fields_for_operators((sink, source))
        strange_edge = peram_incidence._edge_color(quarks[0], antiquarks[0], True)
        light_edge = peram_incidence._edge_color(quarks[1], antiquarks[1], True)
        self.assertIn("s", strange_edge)
        self.assertIn("ud", light_edge)

    def test_compact_signature_preserves_the_descriptive_key_partition(self):
        operators = model.operator_library()["pn"]
        _fields, quarks, antiquarks = model.fields_for_operators(operators)
        q_flavors = tuple(model.flavor_class(field.flavor, True) for field in quarks)
        anti_flavors = tuple(model.flavor_class(field.flavor, True) for field in antiquarks)
        classified_quarks = tuple(
            replace(field, flavor=flavor) for field, flavor in zip(quarks, q_flavors)
        )
        classified_antiquarks = tuple(
            replace(field, flavor=flavor)
            for field, flavor in zip(antiquarks, anti_flavors)
        )
        vocabulary = reducer._FactorGraphVocabulary(
            operators, classified_quarks, classified_antiquarks,
            isospin_degenerate=True,
        )
        descriptive_partition = {}
        compact_partition = {}
        for index, term in enumerate(core.enumerate_wicks(quarks, antiquarks), 1):
            descriptive, _ = reducer.canonical_factor_graph_key(
                operators, quarks, antiquarks, term,
                classified_quarks, classified_antiquarks,
            )
            compact, witness = reducer.canonical_factor_graph_key(
                operators, quarks, antiquarks, term,
                classified_quarks, classified_antiquarks,
                vocabulary=vocabulary, compact=True,
            )
            compact_without_map, no_witness = reducer.canonical_factor_graph_key(
                operators, quarks, antiquarks, term,
                classified_quarks, classified_antiquarks,
                vocabulary=vocabulary, compact=True, include_operator_map=False,
            )
            self.assertIsInstance(compact, bytes)
            self.assertEqual(compact_without_map, compact)
            self.assertEqual(no_witness, ())
            descriptive_partition.setdefault(descriptive, []).append(index)
            compact_partition.setdefault(compact, []).append(index)
            self.assertEqual(len(witness), len(operators))
        self.assertEqual(
            sorted(sorted(group) for group in descriptive_partition.values()),
            sorted(sorted(group) for group in compact_partition.values()),
        )

    def test_pairing_codec_uses_the_smallest_supported_integer_width(self):
        cases = (
            (1, 1),
            (1 << 8, 1),
            ((1 << 8) + 1, 2),
            (1 << 16, 2),
            ((1 << 16) + 1, 4),
            ((1 << 32) + 1, 8),
        )
        for size, expected_width in cases:
            with self.subTest(size=size):
                width = reducer._pairing_width(size)
                self.assertEqual(width, expected_width)
                last_index = size - 1
                self.assertEqual(
                    reducer._unpack_pairing(
                        reducer._pack_pairing((last_index,), width), width,
                    ),
                    (last_index,),
                )

    def test_generic_colored_graph_path_builds_map_only_when_requested(self):
        operators = list(model.operator_library()["pn"])
        operators[0] = replace(operators[0], gamma="gamma_test")
        operators = tuple(operators)
        _fields, quarks, antiquarks = model.fields_for_operators(operators)
        q_flavors = tuple(model.flavor_class(field.flavor, True) for field in quarks)
        anti_flavors = tuple(model.flavor_class(field.flavor, True) for field in antiquarks)
        classified_quarks = tuple(
            replace(field, flavor=flavor) for field, flavor in zip(quarks, q_flavors)
        )
        classified_antiquarks = tuple(
            replace(field, flavor=flavor)
            for field, flavor in zip(antiquarks, anti_flavors)
        )
        self.assertGreater(
            len({reducer._operator_color(operator, True) for operator in operators}), 2,
        )
        vocabulary = reducer._FactorGraphVocabulary(
            operators, classified_quarks, classified_antiquarks,
            isospin_degenerate=True,
        )

        term = next(core.enumerate_wicks(quarks, antiquarks))
        descriptive, witness = reducer.canonical_factor_graph_key(
            operators, quarks, antiquarks, term,
            classified_quarks, classified_antiquarks,
        )
        compact, no_witness = reducer.canonical_factor_graph_key(
            operators, quarks, antiquarks, term,
            classified_quarks, classified_antiquarks,
            vocabulary=vocabulary, compact=True, include_operator_map=False,
        )
        self.assertIsInstance(compact, bytes)
        self.assertEqual(no_witness, ())
        self.assertEqual(len(witness), len(operators))
        self.assertEqual(len(set(witness)), len(operators))

    def test_compact_wick_bucket_round_trips_pairings_and_signs(self):
        operators = model.operator_library()["ppn"]
        _fields, quarks, antiquarks = model.fields_for_operators(operators)
        width = reducer._pairing_width(len(antiquarks))
        bucket = reducer.CompactWickBucket(width, len(quarks))
        original = list(core.enumerate_wicks(quarks, antiquarks))[:9]
        for index, term in enumerate(original, 1):
            bucket.append(index, term)
        self.assertEqual(len(bucket), len(original))
        self.assertEqual(
            len(bucket.pairing_data),
            len(original) * len(quarks) * width,
        )
        for record, term in zip(bucket, original):
            restored = reducer._term_from_compact(record, width)
            self.assertEqual(restored, term)

    def test_contraction_metadata_participates_in_canonical_key(self):
        sink = model.make_operator(
            "sink", "baryon", "p", "sink", ("u", "u", "d"),
            "Cgamma5", (1, 2), open_slot=0,
        )
        source = model.make_operator(
            "source", "baryon", "p", "source", ("u", "u", "d"),
            "Cgamma5_dagger", (1, 2), open_slot=0,
        )
        species_changed = (replace(sink, species="n"), source)
        projector_changed = (
            model.make_operator(
                "sink", "baryon", "p", "sink", ("u", "u", "d"),
                "Cgamma5", (1, 2), open_slot=0, projector="P+",
            ),
            source,
        )

        def key_for(items):
            _fields, qs, bars = model.fields_for_operators(items)
            term = next(core.enumerate_wicks(qs, bars))
            q_classes = tuple(model.flavor_class(field.flavor, True) for field in qs)
            anti_classes = tuple(model.flavor_class(field.flavor, True) for field in bars)
            classified_qs = tuple(
                replace(field, flavor=flavor) for field, flavor in zip(qs, q_classes)
            )
            classified_bars = tuple(
                replace(field, flavor=flavor) for field, flavor in zip(bars, anti_classes)
            )
            return reducer.canonical_factor_graph_key(
                items, qs, bars, term, classified_qs, classified_bars,
            )[0]

        self.assertEqual(key_for((sink, source)), key_for(species_changed))
        self.assertNotEqual(key_for((sink, source)), key_for(projector_changed))


if __name__ == "__main__":
    unittest.main()
