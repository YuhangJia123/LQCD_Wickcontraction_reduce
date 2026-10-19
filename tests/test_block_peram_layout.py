"""Normal perambulator uses (color, spin, color, spin) axes throughout."""

import unittest

import numpy as np

from kernels.block_reduction import (
    BlockDefinition,
    BlockPlan,
    Factor,
    Network,
    ReconstructionTerm,
    TensorSpec,
    evaluate_plan,
)


class NormalPeramLayoutTests(unittest.TestCase):
    def test_w_generation_uses_color_spin_joint_ports(self):
        peram = TensorSpec(
            "S",
            (3, 4, 3, 4),
            ("color", "spin", "color", "spin"),
            kind="peram",
            mode="normal",
            peram_ports=((0, 1), (2, 3)),
        )
        wedge_shape = (3, 4) * 4
        wedge = TensorSpec(
            "W",
            wedge_shape,
            ("color", "spin") * 4,
            kind="wedge",
            mode="normal",
        )
        network = Network((Factor("W", tuple("abcdefgh")),), tuple("abcdefgh"))
        plan = BlockPlan(
            (BlockDefinition("W", "S", ((0, 1), (2, 3)), wedge_shape),),
            (network,),
            (ReconstructionTerm(0, 1, tuple(range(8))),),
            {"S": peram, "W": wedge},
            (),
            (),
            {"output_shape": wedge_shape},
        )
        rng = np.random.default_rng(20261014)
        s = rng.normal(size=peram.shape) + 1j * rng.normal(size=peram.shape)
        actual = evaluate_plan(plan, {"S": s})

        # Flatten each (V, D) port, then apply the fixed antisymmetric block
        # W_(a,b;i,j)=S_(a,i)S_(b,j)-S_(a,j)S_(b,i).
        matrix = s.reshape(12, 12)
        expected = (
            matrix[:, None, :, None] * matrix[None, :, None, :]
            - matrix[:, None, None, :] * matrix[None, :, :, None]
        ).reshape(wedge_shape)
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(actual.shape, wedge_shape)


if __name__ == "__main__":
    unittest.main()
