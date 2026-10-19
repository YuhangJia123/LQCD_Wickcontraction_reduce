"""Independent numeric witnesses for signed graph canonicalization."""

from dataclasses import replace
from fractions import Fraction
import itertools
import unittest
from unittest.mock import patch

import numpy as np
import opt_einsum

import kernels.block_reduction.canonical as canonical_module
from kernels.block_reduction.canonical import (
    OrbitCanonicalizer, canonicalization_cache, canonicalize, canonicalize_view,
    clear_cache,
)
from kernels.block_reduction.model import (
    Factor, GaussianRational, Network, TensorSpec, TensorSymmetry,
    compose_transposes, exact_scalar,
)


def _value(network, arrays):
    labels = {}
    operands = []
    for factor in network.factors:
        operands.extend((arrays[factor.tensor], [labels.setdefault(i, len(labels)) for i in factor.indices]))
    operands.append([labels[i] for i in network.output])
    return np.asarray(opt_einsum.contract(*operands, optimize="greedy"))


class SignedCanonicalTests(unittest.TestCase):
    def assertWitness(self, network, specs, arrays, *, fixed_output):
        result = canonicalize(network, specs, fixed_output=fixed_output)
        actual = _value(network, arrays)
        reconstructed = result.scalar * _value(result.network, arrays).transpose(result.permutation)
        self.assertEqual(actual.shape, reconstructed.shape)
        np.testing.assert_allclose(actual, reconstructed, rtol=1e-12, atol=1e-12)
        return result

    def test_open_antisymmetric_tensor_remains_nonzero_in_free_output_mode(self):
        spec = TensorSpec("A", (4, 4), ("spin",) * 2, symmetries=(TensorSymmetry((1, 0), -1),))
        value = np.arange(16).reshape(4, 4)
        value = value - value.T
        specs, arrays = {"A": spec}, {"A": value}
        original = Network.from_einsum("ab->ab", ("A",))
        swapped = Network.from_einsum("ba->ab", ("A",))
        fixed = self.assertWitness(original, specs, arrays, fixed_output=True)
        fixed_swapped = self.assertWitness(swapped, specs, arrays, fixed_output=True)
        free = self.assertWitness(original, specs, arrays, fixed_output=False)
        free_swapped = self.assertWitness(swapped, specs, arrays, fixed_output=False)
        self.assertEqual(fixed.key, fixed_swapped.key)
        self.assertEqual(fixed.scalar, -fixed_swapped.scalar)
        self.assertEqual(free.key, free_swapped.key)
        self.assertNotEqual(free.scalar, 0)
        self.assertNotEqual(free_swapped.scalar, 0)

    def test_exact_zero_requires_an_odd_automorphism_with_fixed_open_axes(self):
        rng = np.random.default_rng(22)
        raw = rng.normal(size=(4, 4))
        antisymmetric, symmetric = raw - raw.T, raw + raw.T
        specs = {
            "A": TensorSpec("A", (4, 4), ("spin",) * 2, symmetries=(TensorSymmetry((1, 0), -1),)),
            "B": TensorSpec("B", (4, 4), ("spin",) * 2, symmetries=(TensorSymmetry((1, 0), 1),)),
        }
        arrays = {"A": antisymmetric, "B": symmetric}
        contraction = Network.from_einsum("ab,ab->", ("A", "B"))
        for fixed in (False, True):
            result = self.assertWitness(contraction, specs, arrays, fixed_output=fixed)
            self.assertEqual(result.scalar, 0)
        diagonal = Network.from_einsum("aa->", ("A",))
        self.assertEqual(self.assertWitness(diagonal, specs, arrays, fixed_output=True).scalar, 0)
        even_specs = dict(specs, B=replace(specs["B"], symmetries=specs["A"].symmetries))
        even_arrays = dict(arrays, B=antisymmetric)
        self.assertNotEqual(self.assertWitness(contraction, even_specs, even_arrays, fixed_output=True).scalar, 0)

    def test_full_spin_w_uses_joint_ports_and_survives_open_axis_quotient(self):
        rng = np.random.default_rng(20261014)
        matrix = rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12))
        w = (np.einsum("ai,bj->abij", matrix, matrix) - np.einsum("aj,bi->abij", matrix, matrix)).reshape((4, 3) * 4)
        spec = TensorSpec("W", (4, 3) * 4, ("spin", "color") * 4, kind="wedge", symmetries=(
            TensorSymmetry((2, 3, 0, 1, 4, 5, 6, 7), -1),
            TensorSymmetry((0, 1, 2, 3, 6, 7, 4, 5), -1),
        ))
        specs, arrays = {"W": spec}, {"W": w}
        network = Network.from_einsum("abcdefgh->abcdefgh", ("W",))
        result = self.assertWitness(network, specs, arrays, fixed_output=False)
        self.assertNotEqual(result.scalar, 0)
        partial_repeat = Network.from_einsum("abadijkl->abdijkl", ("W",))
        self.assertNotEqual(self.assertWitness(partial_repeat, specs, arrays, fixed_output=True).scalar, 0)
        joint_repeat = Network.from_einsum("ababijkl->abijkl", ("W",))
        self.assertEqual(self.assertWitness(joint_repeat, specs, arrays, fixed_output=True).scalar, 0)

    def test_noninvolutive_six_axis_transpose_witness_has_the_correct_direction(self):
        array = np.arange(4 ** 6).reshape((4,) * 6)
        spec = TensorSpec("X", array.shape, ("spin",) * 6)
        network = Network.from_einsum("abcdef->bcdefa", ("X",))
        result = self.assertWitness(network, {"X": spec}, {"X": array}, fixed_output=False)
        self.assertNotEqual(compose_transposes(result.permutation, result.permutation), tuple(range(6)))
        plain = Network.from_einsum("abcdef->abcdef", ("X",))
        self.assertEqual(result.key, canonicalize(plain, {"X": spec}, fixed_output=False).key)
        self.assertNotEqual(canonicalize(network, {"X": spec}).key, canonicalize(plain, {"X": spec}).key)

    def test_free_output_preserves_mixed_axis_shapes_in_reconstruction(self):
        array = np.arange(4 * 3 * 4).reshape(4, 3, 4)
        specs = {"X": TensorSpec("X", array.shape, ("spin", "color", "spin"))}
        keys = set()
        for output in ("abc", "bca", "cab", "cba"):
            network = Network.from_einsum(f"abc->{output}", ("X",))
            keys.add(self.assertWitness(network, specs, {"X": array}, fixed_output=False).key)
        self.assertEqual(len(keys), 1)

    def test_refinement_does_not_identify_distinct_regular_graphs(self):
        spec = TensorSpec("M", (4, 4), ("spin",) * 2)
        cycle = Network.from_einsum("ab,bc,cd,de,ef,fa->", ("M",) * 6)
        triangles = Network.from_einsum("ab,bc,ca,de,ef,fd->", ("M",) * 6)
        key = canonicalize(cycle, {"M": spec}).key
        self.assertNotEqual(key, canonicalize(triangles, {"M": spec}).key)
        rng = np.random.default_rng(29)
        arrays = {"M": rng.normal(size=(4, 4))}
        self.assertWitness(cycle, {"M": spec}, arrays, fixed_output=True)
        for order in itertools.islice(itertools.permutations(range(6)), 16):
            renaming = dict(zip("abcdef", rng.permutation(tuple("uvwxyz"))))
            renamed = Network(tuple(Factor("M", tuple(renaming[i] for i in cycle.factors[position].indices)) for position in order), ())
            result = self.assertWitness(renamed, {"M": spec}, arrays, fixed_output=True)
            self.assertEqual(result.key, key)

    def test_inconsistent_signed_generator_closure_proves_zero(self):
        # A negative order-three symmetry implies a negative identity.
        spec = TensorSpec("Z", (4,) * 3, ("spin",) * 3, symmetries=(TensorSymmetry((1, 2, 0), -1),))
        network = Network.from_einsum("abc->abc", ("Z",))
        result = self.assertWitness(network, {"Z": spec}, {"Z": np.zeros(spec.shape)}, fixed_output=False)
        self.assertEqual(result.scalar, 0)

    def test_color_epsilon_aliases_and_equivalent_symmetry_generators_share_a_key(self):
        epsilon = np.zeros((3, 3, 3))
        for p in itertools.permutations(range(3)):
            inversions = sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3))
            epsilon[p] = (-1) ** inversions
        spec = TensorSpec("E", (3,) * 3, ("color",) * 3, identity="color-epsilon", symmetries=(
            TensorSymmetry((1, 0, 2), -1), TensorSymmetry((0, 2, 1), -1),
        ))
        alias = replace(spec, name="Alias", symmetries=(
            TensorSymmetry((1, 2, 0), 1), TensorSymmetry((1, 0, 2), -1),
        ))
        specs, arrays = {"E": spec, "Alias": alias}, {"E": epsilon, "Alias": epsilon}
        original = Network.from_einsum("abc->abc", ("E",))
        odd = Network.from_einsum("bac->abc", ("Alias",))
        result = self.assertWitness(original, specs, arrays, fixed_output=True)
        alias_result = self.assertWitness(odd, specs, arrays, fixed_output=True)
        self.assertEqual(result.key, alias_result.key)
        self.assertEqual(result.scalar, -alias_result.scalar)
        self.assertNotEqual(self.assertWitness(original, specs, arrays, fixed_output=False).scalar, 0)

    def test_output_orbit_views_match_legacy_fixed_classes_and_signed_values(self):
        rng = np.random.default_rng(52)
        raw = rng.normal(size=(4, 4))
        antisymmetric = raw - raw.T
        specs = {"A": TensorSpec("A", (4, 4), ("spin",) * 2,
                                 symmetries=(TensorSymmetry((1, 0), -1),))}
        arrays = {"A": antisymmetric}
        orbit = OrbitCanonicalizer(specs)
        networks = [Network.from_einsum(equation, ("A", "A")) for equation in
                    ("ab,cd->abcd", "ab,cd->bacd", "ab,cd->cdab", "ab,cd->acbd", "wx,yz->xwzy")]
        fixed, legacy = [], []
        for network in networks:
            result = orbit.canonicalize(network)
            np.testing.assert_allclose(_value(network, arrays), result.scalar * _value(result.network, arrays),
                                       rtol=1e-12, atol=1e-12)
            fixed.append(result)
            legacy.append(canonicalize(network, specs))
            free = orbit.canonicalize(network, fixed_output=False)
            np.testing.assert_allclose(_value(network, arrays),
                                       free.scalar * _value(free.network, arrays).transpose(free.permutation),
                                       rtol=1e-12, atol=1e-12)
        for i, j in itertools.product(range(len(networks)), repeat=2):
            self.assertEqual(fixed[i].key == fixed[j].key, legacy[i].key == legacy[j].key)
            if fixed[i].key == fixed[j].key:
                self.assertEqual(fixed[i].scalar * fixed[j].scalar, legacy[i].scalar * legacy[j].scalar)

    def test_output_orbit_reuses_free_graph_and_keeps_six_axis_composition(self):
        array = np.arange(4 ** 6).reshape((4,) * 6)
        specs = {"X": TensorSpec("X", array.shape, ("spin",) * 6)}
        original = Network.from_einsum("abcdef->bcdefa", ("X",))
        base = canonicalize(original, specs, fixed_output=False)
        p = (2, 3, 4, 5, 0, 1)
        view = canonicalize_view(base, p, fixed_output=True)
        np.testing.assert_array_equal(_value(original, {"X": array}).transpose(p),
                                      view.scalar * _value(view.network, {"X": array}))
        orbit = OrbitCanonicalizer(specs)
        with patch.object(canonical_module, "canonicalize", wraps=canonicalize) as full_search:
            for output in ("abcdef", "bcdefa", "fedcba", "cdefab"):
                network = Network.from_einsum(f"abcdef->{output}", ("X",))
                fixed = orbit.canonicalize(network)
                orbit.canonicalize(fixed.network, fixed_output=False)
            self.assertEqual(full_search.call_count, 1)

    def test_output_orbit_keeps_complete_signed_six_axis_automorphisms(self):
        rng = np.random.default_rng(98)
        raw = rng.normal(size=(4, 4))
        arrays = {"A": raw - raw.T}
        specs = {"A": TensorSpec("A", (4, 4), ("spin",) * 2,
                                 symmetries=(TensorSymmetry((1, 0), -1),))}
        original = Network.from_einsum("ab,cd,ef->abcdef", ("A",) * 3)
        base = canonicalize(original, specs, fixed_output=False)
        # Three commuting antisymmetric factors: (C2)^3 semidirect S3.
        self.assertEqual(len(base.output_symmetries), 48)
        canonical_value = _value(base.network, arrays)
        for q, sign in base.output_symmetries:
            np.testing.assert_allclose(canonical_value.transpose(q), sign * canonical_value,
                                       rtol=1e-12, atol=1e-12)
        orbit = OrbitCanonicalizer(specs)
        fixed = orbit.canonicalize(original)
        for p in ((2, 3, 4, 5, 0, 1), (3, 2, 4, 5, 0, 1), (1, 0, 3, 2, 5, 4)):
            viewed = Network(original.factors, tuple(original.output[i] for i in p))
            result = orbit.canonicalize(viewed)
            self.assertEqual(result.key, fixed.key)
            np.testing.assert_allclose(_value(viewed, arrays), result.scalar * _value(result.network, arrays),
                                       rtol=1e-12, atol=1e-12)
            for q, sign in result.output_symmetries:
                value = _value(result.network, arrays)
                np.testing.assert_allclose(value.transpose(q), sign * value,
                                           rtol=1e-12, atol=1e-12)

    def test_output_orbit_keeps_typed_zero_shapes_and_open_antisymmetry(self):
        specs = {"A": TensorSpec("A", (4, 4), ("spin",) * 2,
                                 symmetries=(TensorSymmetry((1, 0), -1),)),
                 "X": TensorSpec("X", (4, 3), ("spin", "color"))}
        orbit = OrbitCanonicalizer(specs)
        open_tensor = orbit.canonicalize(Network.from_einsum("ab->ab", ("A",)))
        self.assertNotEqual(open_tensor.scalar, 0)
        zero = orbit.canonicalize(Network.from_einsum("aa,bc->bc", ("A", "X")))
        transposed_zero = orbit.canonicalize(Network.from_einsum("aa,bc->cb", ("A", "X")))
        self.assertEqual(zero.scalar, 0)
        self.assertEqual(transposed_zero.scalar, 0)
        self.assertNotEqual(zero.key, transposed_zero.key)

    def test_scoped_canonical_cache_survives_passes_and_exits_cleanly(self):
        spec = TensorSpec("X", (4, 4), ("spin",) * 2)
        a = Network.from_einsum("ab->ab", ("X",))
        b = Network.from_einsum("ab->ba", ("X",))
        clear_cache()
        with patch.object(canonical_module, "_canonicalize_cached",
                          wraps=canonical_module._canonicalize_cached) as graph_search:
            with canonicalization_cache(max_entries=2):
                canonicalize(a, {"X": spec})
                canonicalize(b, {"X": spec})
                canonicalize(a, {"X": spec})
                self.assertEqual(graph_search.call_count, 2)
                with canonicalization_cache(max_entries=1):
                    canonicalize(a, {"X": spec})
                canonicalize(a, {"X": spec})
                self.assertEqual(graph_search.call_count, 3)
            canonicalize(a, {"X": spec})
            self.assertEqual(graph_search.call_count, 4)

    def test_bounded_orbit_caches_preserve_signed_and_transposed_witnesses(self):
        rng = np.random.default_rng(713)
        raw = rng.normal(size=(4, 4))
        arrays = {"A": raw - raw.T, "M": rng.normal(size=(4, 4))}
        specs = {
            "A": TensorSpec("A", (4, 4), ("spin", "spin"),
                            symmetries=(TensorSymmetry((1, 0), -1),)),
            "M": TensorSpec("M", (4, 4), ("spin", "spin")),
        }
        networks = [
            Network.from_einsum("ab->ab", ("A",)),
            Network.from_einsum("ab->ba", ("A",)),
            Network.from_einsum("ab,cd->abcd", ("A", "A")),
            Network.from_einsum("ab,cd->bacd", ("A", "A")),
            Network.from_einsum("ab,bc->ac", ("A", "M")),
            Network.from_einsum("ab,bc->ca", ("A", "M")),
            Network.from_einsum("aa->", ("A",)),
            Network.from_einsum("ab->ab", ("M",)),
        ]
        baseline = OrbitCanonicalizer(specs, max_bases=64, max_views=128)
        bounded = OrbitCanonicalizer(specs, max_bases=1, max_views=2)
        for network in networks + list(reversed(networks)):
            for fixed_output in (False, True):
                expected = baseline.canonicalize(network, fixed_output=fixed_output)
                result = bounded.canonicalize(network, fixed_output=fixed_output)
                self.assertEqual(result, expected)
                actual = _value(network, arrays)
                reconstructed = result.scalar * _value(result.network, arrays).transpose(result.permutation)
                np.testing.assert_allclose(actual, reconstructed, rtol=1e-12, atol=1e-12)
                self.assertLessEqual(len(bounded._bases), 1)
                self.assertLessEqual(len(bounded._free_results), 2)

    def test_orbit_cache_capacity_is_validated(self):
        specs = {"X": TensorSpec("X", (4,), ("spin",))}
        for option in ({"max_bases": 0}, {"max_views": -1},
                       {"max_bases": True}, {"max_views": 1.5}):
            with self.assertRaises(ValueError):
                OrbitCanonicalizer(specs, **option)


class ExactCoefficientTests(unittest.TestCase):
    def test_exact_complex_hash_matches_equal_python_numbers(self):
        for value in (1 + 1j, -2 - 3j, 0.1 + 1e-30j, 1j, -1 + 0j):
            rational = exact_scalar(value)
            self.assertEqual(rational, value)
            self.assertEqual(hash(rational), hash(value))
            self.assertEqual({rational: "present"}[value], "present")

    def test_direct_complex_components_are_exact_and_tiny_residuals_survive(self):
        value = GaussianRational(0.1, 1e-30)
        self.assertEqual(value.real, Fraction(0.1))
        self.assertEqual(value.imag, Fraction(1e-30))
        remainder = exact_scalar(1 + 1e-30j) + exact_scalar(-1)
        self.assertTrue(remainder)
        self.assertEqual(remainder.real, 0)
        self.assertEqual(remainder.imag, Fraction(1e-30))
        self.assertFalse(value + -value)
        self.assertNotEqual(GaussianRational(1, 0), "1")
        for invalid in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                GaussianRational(invalid, 0)

    def test_axis_metadata_rejects_noninteger_dimensions_and_permutations(self):
        with self.assertRaises(ValueError):
            TensorSymmetry((0.0, 1.0))
        with self.assertRaises(ValueError):
            TensorSpec("X", (4.0,), ("spin",))
        with self.assertRaises(ValueError):
            TensorSpec("S", (4, 3, 4, 3), ("spin", "color") * 2, kind="peram", peram_ports=((0.0, 1), (2, 3)))
        spec = TensorSpec("X", (np.int64(4),), ("spin",))
        self.assertEqual(spec.shape, (4,))


if __name__ == "__main__":
    unittest.main()
