"""Cycle-type Burnside counts checked against explicit small group actions."""

from itertools import combinations, permutations
import unittest

from kernels.block_reduction.pairing_count import count_pairing_layout


def _complete_family(sink_types, source_types):
    return tuple(pairing for pairing in permutations(range(len(sink_types)))
                 if all(sink_types[i] == source_types[pairing[i]]
                        for i in range(len(pairing))))


def _brute_orbit_count(sink_types, source_types, pairs, generators):
    remaining = set(_complete_family(sink_types, source_types))
    size = len(sink_types)
    identity = tuple(range(size))
    actions = list(generators) + [
        (tuple(right if i == left else left if i == right else i
               for i in range(size)), identity)
        for left, right in pairs
    ]
    orbits = 0
    while remaining:
        seed = next(iter(remaining))
        orbit = {seed}
        pending = [seed]
        while pending:
            pairing = pending.pop()
            for sink, source in actions:
                transformed = [None] * size
                for old, value in enumerate(pairing):
                    transformed[sink[old]] = source[value]
                transformed = tuple(transformed)
                if transformed not in orbit:
                    orbit.add(transformed)
                    pending.append(transformed)
        remaining.difference_update(orbit)
        orbits += 1
    return orbits


class PairingCountTests(unittest.TestCase):
    def assert_count_matches_brute(self, sink_types, source_types, pairs, generators=()):
        raw = _complete_family(sink_types, source_types)
        result = count_pairing_layout(sink_types, source_types, pairs, generators,
                                      pairings=raw)
        self.assertTrue(result.full_family_verified)
        self.assertEqual(result.pairing_count, len(raw))
        self.assertEqual(result.orbit_count,
                         _brute_orbit_count(sink_types, source_types, pairs,
                                            generators))

    def test_small_complete_families_and_source_action(self):
        for count in range(2, 6):
            types = ("u",) * count
            with self.subTest(count=count):
                self.assert_count_matches_brute(types, types, ((0, 1),))
                source_swap = (tuple(range(count)),
                               (1, 0) + tuple(range(2, count)))
                self.assert_count_matches_brute(types, types, ((0, 1),),
                                                (source_swap,))

    def test_two_block_cycle_and_mixed_classes(self):
        types = ("u", "u", "u", "u", "d", "d")
        identity = tuple(range(6))
        block_swap = ((2, 3, 0, 1, 4, 5), identity)
        source_swap = (identity, (1, 0, 2, 3, 4, 5))
        self.assert_count_matches_brute(types, types,
                                        ((0, 1), (2, 3), (4, 5)),
                                        (block_swap, source_swap))

    def test_unstable_generator_is_filtered_to_layout_stabilizer(self):
        types = ("u",) * 4
        identity = tuple(range(4))
        cycle = ((1, 2, 3, 0), identity)
        result = count_pairing_layout(types, types, ((0, 1), (2, 3)),
                                      (cycle,))
        self.assertEqual(result.layout_group_order, 2)
        # Brute force only uses the subgroup preserving the matching.
        stable = (((2, 3, 0, 1), identity),)
        self.assertEqual(result.orbit_count,
                         _brute_orbit_count(types, types,
                                            ((0, 1), (2, 3)), stable))

    def test_noninvolutive_three_block_cycle(self):
        types = ("u",) * 6
        identity = tuple(range(6))
        cycle = (2, 3, 4, 5, 0, 1)
        self.assert_count_matches_brute(
            types, types, ((0, 1), (2, 3), (4, 5)),
            ((cycle, identity), (identity, cycle)),
        )

    def test_fixed_point_sum_with_no_one_cycles(self):
        types = ("u",) * 4
        identity = tuple(range(4))
        source_swap = (identity, (1, 0, 3, 2))
        self.assert_count_matches_brute(
            types, types, ((0, 1), (2, 3)), (source_swap,),
        )

    def test_known_nine_slot_layout_histogram_is_conditional(self):
        types = ("u", "u", "d", "u", "u", "d", "d", "d", "u")
        identity = tuple(range(9))
        whole_swap = (3, 4, 5, 0, 1, 2, 6, 7, 8)
        generators = ((whole_swap, identity), (identity, whole_swap))
        d_pairs = (((2, 5), (6, 7)), ((2, 6), (5, 7)),
                   ((2, 7), (5, 6)))
        histogram = {}
        for quartet in combinations((0, 1, 3, 4, 8), 4):
            a, b, c, d = quartet
            for u_pairs in (((a, b), (c, d)), ((a, c), (b, d)),
                            ((a, d), (b, c))):
                for d_pairing in d_pairs:
                    result = count_pairing_layout(
                        types, types, u_pairs + d_pairing, generators)
                    histogram[result.orbit_count] = histogram.get(
                        result.orbit_count, 0) + 1
                    self.assertEqual(result.pairing_count, 2880)
                    self.assertFalse(result.full_family_verified)
        self.assertEqual(histogram, {48: 2, 92: 43})

    def test_rejects_incomplete_family_or_illegal_actions(self):
        types = ("u", "u", "d")
        with self.assertRaisesRegex(ValueError, "not complete"):
            count_pairing_layout(types, types, ((0, 1),), pairings=((0, 1, 2),))
        with self.assertRaisesRegex(ValueError, "pairing class"):
            count_pairing_layout(types, types, ((0, 2),))
        with self.assertRaisesRegex(ValueError, "pairing class"):
            count_pairing_layout(types, types, ((0, 1),),
                                 (((2, 1, 0), (0, 1, 2)),))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            raw = _complete_family(types, types)
            count_pairing_layout(types, types, ((0, 1),),
                                 pairings=raw + raw[:1])
        with self.assertRaisesRegex(ValueError, "permutation"):
            count_pairing_layout(types, types, (),
                                 (((0.0, 1.0, 2.0), (0, 1, 2)),))


if __name__ == "__main__":
    unittest.main()
