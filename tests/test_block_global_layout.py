"""Real pairing-family layout search through the independent tensor layer."""

from collections import defaultdict
import unittest

from kernels.block_reduction.canonical import OrbitCanonicalizer
from kernels.block_reduction.global_layout import PairingFamily, find_pairing_layouts
from kernels.block_reduction.model import Network, WeightedNetwork, exact_scalar
from kernels.block_reduction.planner import _block_specs
from verify_code.normal_block_validation import tensor_environment
from verify_code.ppn_global_w_basis_audit import Case


class GlobalLayoutTests(unittest.TestCase):
    def test_ppn_discovers_both_48_network_layouts(self):
        case = Case()
        first_equation, names, _output = case.equation(case.terms[0].pairing)
        source = "x=contract(" + repr(first_equation) + "," + ",".join(names) + ")"
        specs, _constants = tensor_environment(source)
        positions = tuple(i for i, name in enumerate(names) if name.startswith("peram_"))
        family = PairingFamily(tuple(
            WeightedNetwork(Network.from_einsum(*case.equation(term.pairing)[:2]), term.sign)
            for term in case.terms
        ), positions)

        result = find_pairing_layouts(family, specs, search_seconds=5.0, top_k=2)

        self.assertEqual(len(family.terms), 2880)
        self.assertEqual(result.sink_types.count(result.sink_types[0]), 5)
        self.assertEqual(len(result.endpoint_witnesses), 3)
        self.assertEqual({(item.output_permutation, item.sign)
                          for item in result.endpoint_witnesses}, {
            ((1, 0, 2, 3, 4, 5), -1),
            ((0, 1, 2, 4, 3, 5), -1),
            ((1, 0, 2, 4, 3, 5), 1),
        })
        self.assertEqual(len(result.layouts), 2)
        self.assertEqual({layout.network_count for layout in result.layouts}, {48})
        self.assertEqual({layout.pairs for layout in result.layouts}, {
            ((0, 1), (2, 5), (3, 4), (6, 7)),
            ((0, 4), (1, 3), (2, 5), (6, 7)),
        })
        for layout in result.layouts:
            self.assertEqual(layout.count.orbit_count, 48)
            self.assertTrue(layout.count.full_family_verified)
            self.assertEqual(len(layout.fused_terms), 180)
            self.assertTrue(all(sum(f.tensor.startswith("W_") for f in term.network.factors) == 4
                                for term in layout.fused_terms))

        # Compare the fast endpoint-orbit score with independent term-by-term
        # fixed-axis canonicalization, including signed coefficient collection.
        expanded_specs, _definitions, _by_prop = _block_specs(specs)
        canonicalizer = OrbitCanonicalizer(expanded_specs)
        for layout in result.layouts:
            ledger = defaultdict(int)
            live = {}
            for term in layout.fused_terms:
                witness = canonicalizer.canonicalize(term.network, fixed_output=True)
                if witness.scalar:
                    ledger[witness.key] += exact_scalar(term.coefficient) * witness.scalar
                    live[witness.key] = witness.network
            exact_count = len({
                canonicalizer.canonicalize(live[key], fixed_output=False).key
                for key, coefficient in ledger.items() if coefficient
            })
            self.assertEqual(layout.network_count, exact_count)


if __name__ == "__main__":
    unittest.main()
