import json
import tempfile
import unittest
from pathlib import Path

from kernels import correlator_wick_normalizer as core
from verify_code.operator_input import enumerate_raw_wicks, load_operators
from kernels.wick_reducer import operator_model
from kernels.wick_reducer import peram_incidence


class OperatorInputTests(unittest.TestCase):
    def test_flat_and_sink_source_inputs_preserve_ordered_fields(self):
        sink = operator_model.make_operator(
            "sink", "baryon", "p", "sink_p", ("u", "u", "d"),
            "Cgamma5", gamma_slots=(1, 2), open_slot=0,
        )
        source = operator_model.make_operator(
            "source", "baryon", "p", "source_p", ("u", "d", "u"),
            "Cgamma5_dagger", gamma_slots=(1, 2), open_slot=0,
            external_tag="t0",
        )
        operators = load_operators([[sink], [source]])
        self.assertEqual(tuple(item.label for item in operators), ("sink_p", "source_p"))
        self.assertEqual(operators[1].external_tag, "t0")

    def test_json_mapping_and_raw_enumeration(self):
        document = {
            "name": "pn",
            "operators": [
                {"side": "sink", "kind": "baryon", "species": "p", "label": "p_s",
                 "flavors": ["u", "u", "d"], "gamma": "G", "gamma_slots": [1, 2], "open_slot": 0},
                {"side": "source", "kind": "baryon", "species": "p", "label": "p_0",
                 "flavors": ["u", "d", "u"], "gamma": "Gd", "gamma_slots": [1, 2], "open_slot": 0},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "correlator.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            operators, _fields, quarks, antiquarks, terms = enumerate_raw_wicks(path)
        self.assertEqual(len(operators), 2)
        self.assertEqual(len(quarks), len(antiquarks), 3)
        self.assertEqual(len(terms), 2)

    def test_contraction_metadata_is_normalized_and_part_of_incidence_color(self):
        first = operator_model.make_operator(
            "sink", "baryon", "p", "p_s", ("u", "u", "d"),
            "Cgamma5", gamma_slots=(1, 2), open_slot=0,
            contraction_metadata={"projector": "P+", "custom": [1, 2]},
            momentum=(0, 0, 1), clebsch=(1, 2), phase=1j,
        )
        reordered = operator_model.make_operator(
            "sink", "baryon", "n", "n_s", ("u", "u", "d"),
            "Cgamma5", gamma_slots=(1, 2), open_slot=0,
            phase=1j, clebsch=(1, 2), momentum=(0, 0, 1),
            contraction_metadata={"custom": (1, 2), "projector": "P+"},
        )
        self.assertEqual(first.contraction_metadata, reordered.contraction_metadata)
        self.assertIsInstance(first.contraction_metadata, tuple)
        self.assertIn("projector", peram_incidence._operator_color(first))
        changed = operator_model.make_operator(
            "sink", "baryon", "p", "p_s", ("u", "u", "d"),
            "Cgamma5", gamma_slots=(1, 2), open_slot=0,
            contraction_metadata={"projector": "P-", "custom": [1, 2]},
            momentum=(0, 0, 1), clebsch=(1, 2), phase=1j,
        )
        self.assertNotEqual(
            peram_incidence._operator_color(first),
            peram_incidence._operator_color(changed),
        )

    def test_json_input_accepts_metadata_aliases_and_legacy_defaults(self):
        document = {
            "operators": [
                {"side": "sink", "kind": "meson", "species": "pi", "label": "pi",
                 "flavors": ["u"], "momentum": [1, 0, 0], "projector": "P+",
                 "clebsch": [1, 2], "phase": [0, 1]},
                {"side": "source", "kind": "meson", "species": "pi", "label": "pi0",
                 "flavors": ["u"]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            operators = load_operators(path)
        self.assertEqual(
            operators[0].contraction_metadata,
            (("clebsch", (1, 2)), ("momentum", (1, 0, 0)),
             ("phase", (0, 1)), ("projector", "P+")),
        )
        self.assertEqual(operators[1].contraction_metadata, ())

    def test_complex_metadata_round_trips_through_normalizer_json(self):
        operator = operator_model.make_operator(
            "sink", "baryon", "p", "p_s", ("u", "d", "u"), phase=1j,
        )
        document = {
            "name": "metadata_roundtrip",
            "operators": [{
                "side": operator.side, "kind": operator.kind,
                "species": operator.species, "label": operator.label,
                "flavors": list(operator.flavors), "gamma": operator.gamma,
                "gamma_slots": list(operator.gamma_slots),
                "open_slot": operator.open_slot,
                "contraction_metadata": core.contraction_metadata_dict(
                    operator.contraction_metadata,
                ),
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            _name, loaded, _matrices = core._from_json(path)
        self.assertEqual(loaded[0].contraction_metadata, operator.contraction_metadata)


if __name__ == "__main__":
    unittest.main()
