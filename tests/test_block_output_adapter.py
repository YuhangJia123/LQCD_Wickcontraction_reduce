"""The project adapter reads real-style normal exports as static data."""

import unittest
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np

from kernels.block_reduction import evaluate_expression
from kernels.block_reduction.output_adapter import load_normal_output


PERAM = "peram_u_sin_sou_x_x"


class NormalOutputAdapterTests(unittest.TestCase):
    def test_gamma_cross_transpose_and_noninvolutive_view(self):
        gamma = np.arange(16).reshape(4, 4).astype(np.complex128)
        gamma[1, 3] += 1j
        source = f"""# Operator mode: normal. All tensor arrays are supplied by the caller.
#   {PERAM}: {{"quark": {{"label": "q"}}, "antiquark": {{"label": "a"}}}}
from opt_einsum import contract
x_basis_contract = {{}}
x_basis_contract[1] = contract('ab,iacB->Bbi', G, {PERAM})
x_basis_contract[2] = contract('ab,iacB->Bbi', H, {PERAM})
x_basis = {{}}
x_basis[0] = x_basis_contract[1].transpose((1, 2, 0))
x_basis[1] = -x_basis_contract[2].transpose((1, 2, 0))
x_coefficients = [3, -2]
"""
        adapted = load_normal_output(source, gamma_arrays={"G": gamma, "H": -gamma.T},
                                     normalize_matrices=True)
        self.assertEqual(adapted.tensor_specs[PERAM].shape, (3, 4, 3, 4))
        self.assertEqual(adapted.peram_metadata[PERAM].quark_side, "sin")
        self.assertEqual(adapted.peram_metadata[PERAM].endpoints["quark"]["label"], "q")
        self.assertIn(("H", "G", (1, 0), -1),
                      {(r.source, r.target, r.permutation, r.sign)
                       for r in adapted.matrix_relations})
        self.assertEqual(adapted.basis[0].permutation, (1, 2, 0))
        arrays = {PERAM: np.random.default_rng(19).standard_normal((3, 4, 3, 4)),
                  "G": gamma, "H": -gamma.T}
        original = adapted.original_symbolic
        expected = evaluate_expression(original.basis_contract, original.basis,
                                       original.coefficients, arrays)
        actual = evaluate_expression(adapted.basis_contract, adapted.basis,
                                     adapted.coefficients, arrays)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_distinct_endpoint_and_isospin_names_require_explicit_alias(self):
        names = ("peram_u_sin_sou_x_x_p_alpha", "peram_u_sin_sou_x_x_p_beta",
                 "peram_d_sin_sou_x_x")
        source = "\n".join([
            "# Operator mode: normal. All tensor arrays are supplied by the caller.",
            f'#   {names[0]}: {{"quark": {{"label": "alpha"}}, "antiquark": {{"label": "a"}}}}',
            f'#   {names[1]}: {{"quark": {{"label": "beta"}}, "antiquark": {{"label": "a"}}}}',
            "from opt_einsum import contract", "x_basis_contract = {}",
            *(f"x_basis_contract[{i}] = contract('iAaB->AB', {name})"
              for i, name in enumerate(names)),
            "x_basis = {}", "x_basis[0] = x_basis_contract[0]",
            "x_basis[1] = x_basis_contract[1]", "x_basis[2] = x_basis_contract[2]",
            "x_coefficients = [1, 1, 1]",
        ])
        default = load_normal_output(source, gamma_arrays={})
        identities = [default.tensor_specs[name].identity for name in names]
        self.assertEqual(identities, list(names))
        self.assertEqual(default.peram_metadata[names[0]].endpoint_tag, "alpha")
        self.assertEqual(default.peram_metadata[names[1]].endpoints["quark"]["label"], "beta")
        aliased = load_normal_output(source, gamma_arrays={},
                                     aliases={names[1]: names[0], names[2]: names[0]})
        self.assertEqual({aliased.tensor_specs[name].identity for name in names}, {names[0]})

    def test_mode_unknown_tensor_and_unproved_symmetry_are_rejected(self):
        source = """# Operator mode: normal. All tensor arrays are supplied by the caller.
from opt_einsum import contract
x_basis_contract = {0: contract('ab->ab', Unknown)}
x_basis = {0: x_basis_contract[0]}
x_coefficients = [1]
"""
        with self.assertRaisesRegex(ValueError, "unknown tensor"):
            load_normal_output(source, gamma_arrays={})
        with self.assertRaisesRegex(ValueError, "normal-mode"):
            load_normal_output(source.replace("# Operator mode: normal.", "# Operator mode: distillation."),
                               gamma_arrays={})
        from kernels.block_reduction import TensorSpec, TensorSymmetry
        false_spec = TensorSpec("Unknown", (4, 4), ("spin", "spin"),
                                symmetries=(TensorSymmetry((1, 0), -1),))
        with self.assertRaisesRegex(ValueError, "unproved tensor symmetry"):
            load_normal_output(source, gamma_arrays={}, extra_specs={"Unknown": false_spec},
                               extra_arrays={"Unknown": np.eye(4)})

    def test_registered_constants_cannot_be_falsely_aliased(self):
        source = """# Operator mode: normal. All tensor arrays are supplied by the caller.
from opt_einsum import contract
x_basis_contract = {0: contract('ab,bc->ac', G, H)}
x_basis = {0: x_basis_contract[0]}
x_coefficients = [1]
"""
        with self.assertRaisesRegex(ValueError, "contradicts the registered arrays"):
            load_normal_output(source, gamma_arrays={"G": np.eye(4), "H": -np.eye(4)},
                               aliases={"H": "G"})

    def test_real_ppn_three_tables_and_optional_raw_family(self):
        from verify_code.normal_block_validation import generate_normal_fixture
        from kernels.block_reduction.output_adapter import pairing_family_from_operators
        from kernels.wick_reducer.operator_model import operator_library

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal_ppn.py"
            generate_normal_fixture("ppn", path)
            adapted = load_normal_output(path)
        self.assertEqual(len(adapted.basis_contract), 219)
        self.assertEqual(len(adapted.basis), 2760)
        self.assertEqual(len(adapted.coefficients), 2760)
        self.assertEqual(adapted.tensor_specs[PERAM].peram_ports, ((0, 1), (2, 3)))
        operators = tuple(replace(operator, mode="normal")
                          for operator in operator_library()["ppn"])
        family = pairing_family_from_operators(operators, tensor_specs=adapted.tensor_specs)
        self.assertEqual(len(family.terms), 2880)
        self.assertEqual(len(family.factor_positions), 9)
        self.assertEqual({term.coefficient for term in family.terms}, {-1, 1})
        distinct_flavors = pairing_family_from_operators(operators, isospin_degenerate=False)
        self.assertEqual(len(distinct_flavors.terms), 2880)
        self.assertEqual({factor.tensor.split("_")[1] for term in distinct_flavors.terms
                          for factor in term.network.factors if factor.tensor.startswith("peram_")},
                         {"u", "d"})


if __name__ == "__main__":
    unittest.main()
