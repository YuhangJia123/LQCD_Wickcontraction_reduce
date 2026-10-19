"""Regression tests for the merged input/mode/output facade."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kernels.wick_colour_orbit.user_api import make_operator
from kernels.wick_reducer.unified import (
    UnifiedConfig,
    canonical_input,
    canonical_output,
    load_unified_input,
    run_unified,
    write_unified,
)


MESON = {
    "side": "sink", "kind": "meson", "species": "pi", "label": "sink_pi",
    "flavors": ["u"], "gamma": "gamma5",
}


class UnifiedFacadeTests(unittest.TestCase):
    def _document(self, **updates):
        source = dict(MESON)
        source["side"] = "source"
        source["label"] = "source_pi"
        value = {
            "name": "pi",
            "input": "normal",
            "mode": "special",
            "output": "independent",
            "operators": [MESON, source],
        }
        value.update(updates)
        return value

    def test_aliases_are_canonicalized(self):
        self.assertEqual(canonical_input("distrilation"), "distillation")
        self.assertEqual(canonical_output("indenpent"), "independent")

    def test_direct_alias_config_and_sibling_operators(self):
        config = UnifiedConfig(input="distrilation", mode="wick", output="indenpent")
        self.assertEqual(config.input, "distillation")
        self.assertEqual(config.output, "independent")
        special = UnifiedConfig(input="normal", mode="special", output="check")
        sink = make_operator("sink", "meson", "pi", "s", ("u",), "gamma5")
        source = make_operator("source", "meson", "pi", "t", ("u",), "gamma5")
        report = run_unified(special, [[sink], [source]])
        self.assertEqual(report["check"]["status"], "passed")

    def test_special_check_and_result_directory(self):
        config, operators, matrices = load_unified_input(self._document(output="check"))
        report = run_unified(config, operators, gamma_matrices=matrices)
        self.assertEqual(report["representation"], "incidence-s3-packed")
        self.assertEqual(report["check"]["status"], "passed")
        with tempfile.TemporaryDirectory() as directory:
            written = write_unified(config, operators, result_root=directory)
            result_path = Path(written["result_path"])
            self.assertEqual(result_path.parent.name, "normal_special_check")
            document = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(document["format"], "wick-colour-orbit-json-v3")
            manifest = json.loads((result_path.parent / "manifest.json").read_text())
            self.assertEqual(manifest["input"], "normal")
            self.assertEqual(manifest["mode"], "special")

    def test_special_scopes_select_sibling_fusion_kernels(self):
        config, operators, matrices = load_unified_input(self._document(output="check"))
        complete = run_unified(config, operators, gamma_matrices=matrices)
        cross_time = run_unified(
            UnifiedConfig(input="normal", mode="special", output="check",
                          name="pi", wick_scope="cross_time"),
            operators, gamma_matrices=matrices,
        )
        self.assertEqual(config.wick_scope, "complete")
        self.assertEqual(complete["stats"]["route"], "general-fused")
        self.assertEqual(cross_time["stats"]["route"], "cross-time-fused")
        self.assertEqual(complete["check"]["status"], "passed")
        self.assertEqual(cross_time["check"]["status"], "passed")

    def test_invalid_combinations_are_rejected(self):
        with self.assertRaises(ValueError):
            load_unified_input(self._document(input="distillation"))
        with self.assertRaises(ValueError):
            load_unified_input(self._document(output="count", mode="wick"))
        with self.assertRaises(ValueError):
            load_unified_input(self._document(output="block"))
        with self.assertRaises(ValueError):
            UnifiedConfig(input="normal", mode="special", output="block")


if __name__ == "__main__":
    unittest.main()
