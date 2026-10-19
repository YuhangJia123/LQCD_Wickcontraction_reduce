import ast
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels.wick_reducer import expression_output
from kernels.wick_reducer.reducer import reduce_operators
from verify_code import check_basis_vs_original


class StreamedCheckOutputTests(unittest.TestCase):
    def test_generic_streamed_check_matches_detailed_format(self):
        operators = expression_output.model.operator_library()["delta"]
        expected_report = reduce_operators(
            operators, label="small_fixture", mode="detailed",
        )
        expected = expression_output.render_python_output(
            expected_report, operators, name="small_fixture", mode="check",
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "check.py"
            result = expression_output.write_output(
                expression_output.split_operators(operators),
                output_path,
                name="small_fixture",
                mode="check",
            )
            actual = output_path.read_text(encoding="utf-8")

        self.assertEqual(actual, expected)
        ast.parse(actual)
        self.assertEqual(result["wick_count"], 6)
        self.assertEqual(result["independent_count"], 2)
        self.assertEqual(result["dependent_count"], 4)
        self.assertTrue(result["report"]["check_output_streamed"])
        self.assertIsNone(result["report"]["graph_records"])

    def test_basis_audit_rehydrates_small_report_for_numeric_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = check_basis_vs_original.run(
                names=("delta",),
                include_ppnn=False,
                dimensions=(3,),
                output_dir=Path(directory),
                numeric=True,
            )
        item = summary["fixtures"][0]
        self.assertEqual(item["name"], "delta")
        self.assertEqual(item["mode"], "detailed")
        self.assertEqual(len(item["numeric"]), 1)
        self.assertTrue(item["numeric"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
