import sys
import unittest
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verify_code import ppnn_group_action as action


class GroupActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.he4_states = tuple(action.he4_states())
        cls.he4_state_count = len(cls.he4_states)
        cls.he4_fixed_points = action.fixed_point_histogram(
            cls.he4_states,
            action.BLOCK_GENERATORS,
            action.isospin_flip,
        )
        cls.he4_orbits = action.orbit_size_histogram(
            cls.he4_states,
            action.HE4_GENERATORS,
        )

    def test_he4_state_space_and_burnside_count(self):
        self.assertEqual(self.he4_state_count, 32400)
        self.assertEqual(len(set(self.he4_states)), self.he4_state_count)
        self.assertEqual(
            self.he4_fixed_points,
            Counter({32400: 1, 576: 1, 180: 4, 96: 4, 24: 4, 0: 18}),
        )

    def test_he4_orbit_sizes_cover_all_states(self):
        self.assertEqual(self.he4_orbits, Counter({32: 967, 16: 83, 8: 14, 4: 4}))
        self.assertEqual(
            sum(size * count for size, count in self.he4_orbits.items()),
            self.he4_state_count,
        )

    def test_he4_generators_form_the_declared_action(self):
        state_set = set(self.he4_states)
        for state in self.he4_states:
            for generator in action.HE4_GENERATORS:
                self.assertIn(generator(state), state_set)
                self.assertEqual(generator(generator(state)), state)

            for left_index, left in enumerate(action.BLOCK_GENERATORS):
                for right in action.BLOCK_GENERATORS[left_index + 1 :]:
                    self.assertEqual(left(right(state)), right(left(state)))

            self.assertEqual(action.isospin_flip(action.isospin_flip(state)), state)
            self.assertEqual(
                action.isospin_flip(action.swap_sink_protons(action.isospin_flip(state))),
                action.swap_sink_neutrons(state),
            )
            self.assertEqual(
                action.isospin_flip(action.swap_source_protons(action.isospin_flip(state))),
                action.swap_source_neutrons(state),
            )

    def test_he3_calibration_gives_93_orbits(self):
        states = tuple(action.he3_states())
        self.assertEqual(len(states), 360)
        orbits = action.orbit_size_histogram(
            states,
            (action.he3_sink_proton_swap, action.he3_source_proton_swap),
        )
        self.assertEqual(sum(orbits.values()), 93)
        self.assertEqual(orbits, Counter({4: 87, 2: 6}))


if __name__ == "__main__":
    unittest.main()
