import unittest

from kernels.block_reduction import (
    BlockDefinition,
    BlockPlan,
    Factor,
    Network,
    ReconstructionTerm,
    TensorSpec,
)
from kernels.block_reduction.io import load_symbolic_output
from kernels.block_reduction.output_adapter import load_normal_output, render_block_output


class BlockOutputFormatTests(unittest.TestCase):
    def test_w_definition_precedes_compatible_expression_tables(self):
        peram = TensorSpec(
            "P", (3, 4, 3, 4), ("color", "spin", "color", "spin"),
            kind="peram", peram_ports=((0, 1), (2, 3)),
        )
        definition = BlockDefinition("W_P", "P", peram.peram_ports)
        wedge = TensorSpec("W_P", (3, 4) * 4, ("color", "spin") * 4,
                           kind="wedge")
        network = Network((Factor("W_P", tuple("abcdefgh")),), tuple("abcdefgh"))
        plan = BlockPlan(
            (definition,), (network,),
            (ReconstructionTerm(0, 1, tuple(range(8))),),
            {"P": peram, "W_P": wedge}, (), (), {"output_shape": (3, 4) * 4},
        )

        text = render_block_output(plan)
        self.assertLess(text.index("W_P = contract"), text.index("normal_wick_block_basis_contract"))
        self.assertIn("normal_wick_block_basis_contract[1] = contract('abcdefgh->abcdefgh', W_P)", text)
        self.assertIn("normal_wick_block_coefficients = [1]", text)

        loaded = load_symbolic_output(text, tensor_specs={"P": peram, "W_P": wedge})
        self.assertEqual(loaded.prefix, "normal_wick_block")
        self.assertEqual(len(loaded.basis_contract), 1)
        self.assertEqual(len(loaded.basis), 1)
        self.assertEqual(loaded.basis[0].contract_id, 1)

    def test_normal_adapter_infers_standard_wedge_definition(self):
        peram_name = "peram_u_sin_sou_x_x"
        peram = TensorSpec(
            peram_name, (3, 4, 3, 4), ("color", "spin", "color", "spin"),
            kind="peram", peram_ports=((0, 1), (2, 3)),
        )
        definition = BlockDefinition("W_peram_u_sin_sou_x_x", peram_name,
                                     peram.peram_ports)
        wedge = TensorSpec(definition.name, (3, 4) * 4,
                           ("color", "spin") * 4, kind="wedge")
        network = Network((Factor(definition.name, tuple("abcdefgh")),),
                          tuple("abcdefgh"))
        plan = BlockPlan(
            (definition,), (network,),
            (ReconstructionTerm(0, 1, tuple(range(8))),),
            {peram_name: peram, definition.name: wedge}, (), (),
            {"output_shape": (4,) * 8},
        )
        adapted = load_normal_output(render_block_output(plan))
        self.assertEqual(adapted.tensor_specs[definition.name].kind, "wedge")
        self.assertEqual(len(adapted.basis_contract), 1)


if __name__ == "__main__":
    unittest.main()
