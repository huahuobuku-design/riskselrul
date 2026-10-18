"""Exercise every archived state in CI without NASA data or a network download.

Numerical windows are synthetic implementation fixtures, never RUL benchmarks.
"""
from pathlib import Path
import unittest
import numpy as np
from riskselrul.deploy import lean


class TestAllArchivedStatesSynthetic(unittest.TestCase):
    def test_all_states_are_usable_without_raw_data(self):
        states = sorted((Path(__file__).resolve().parents[1] / "deploy/states").glob("*.npz"))
        self.assertEqual(len(states), 78, "The reviewed state collection must be present")
        for path in states:
            with self.subTest(state=path.name):
                state = lean.load(path)
                raw = np.broadcast_to(state["raw_MU21"], (2, 30, 21)).copy()
                raw[1, 8:, 2] = np.nan
                p, q = lean.predict(raw, state)
                self.assertEqual(p.shape, (2,))
                self.assertEqual(q.shape, (2,))
                self.assertTrue(np.isfinite(p).all() and np.isfinite(q).all())
                self.assertTrue((p >= 0).all())


if __name__ == "__main__":
    unittest.main()
