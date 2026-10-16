"""Lean deployment states must load and run from raw windows, deterministically.

Skipped automatically when the released states or the C-MAPSS text files are not
present. Override the locations with ``RISKSELRUL_STATES`` and
``RISKSELRUL_DATA_ROOT``.
"""

import os
import unittest

import numpy as np

from riskselrul.deploy import lean

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STATES = os.environ.get("RISKSELRUL_STATES", os.path.join(REPO, "deploy", "states"))
DATA = os.environ.get("RISKSELRUL_DATA_ROOT", os.path.join(REPO, "data", "raw"))


def _first_state(suffix="linear_57.npz"):
    if not os.path.isdir(STATES):
        return None
    for f in sorted(os.listdir(STATES)):
        if f.endswith(suffix):
            return os.path.join(STATES, f)
    return None


def _raw_windows(ds="FD001", n=4):
    """Tail-of-test windows for the first ``n`` engines, shape ``(n, 30, 21)``.

    ``test_*.txt`` has 26 columns (unit, cycle, op1..op3, s1..s21); the last 21
    columns are the sensor channels expected by the lean states.
    """
    te = np.loadtxt(os.path.join(DATA, f"test_{ds}.txt"))
    out = []
    for u in range(1, n + 1):
        a = te[te[:, 0] == u]
        out.append(a[-30:, 5:])
    return np.array(out, float)


class TestLeanForward(unittest.TestCase):
    def setUp(self):
        path = _first_state()
        if path is None or not os.path.isfile(os.path.join(DATA, "test_FD001.txt")):
            self.skipTest("released states or C-MAPSS data not available")
        self.state = lean.load(path)
        self.raw = _raw_windows("FD001", 4)

    def test_shapes_and_finiteness(self):
        p, q = lean.predict(self.raw, self.state)
        self.assertEqual(p.shape, (4,))
        self.assertEqual(q.shape, (4,))
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertTrue(np.all(p >= 0))  # non-negative clipping

    def test_deterministic(self):
        p1, q1 = lean.predict(self.raw, self.state)
        p2, q2 = lean.predict(self.raw, self.state)
        self.assertTrue(np.array_equal(p1, p2))
        self.assertTrue(np.array_equal(q1, q2))

    def test_decide_is_boolean_and_uses_frozen_threshold(self):
        p, q, acc = lean.decide(self.raw, self.state, "C2", 0.8)
        self.assertEqual(acc.dtype, bool)
        ix = np.where((self.state["threshold_condition"] == "C2")
                      & (self.state["threshold_coverage"] == 0.8))[0]
        self.assertEqual(len(ix), 1)
        self.assertTrue(np.array_equal(acc, q <= self.state["threshold_value"][ix[0]]))

    def test_nan_tolerant(self):
        raw = self.raw.copy()
        raw[0, 5:, 2] = np.nan
        p, q = lean.predict(raw, self.state)
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertTrue(np.all(np.isfinite(q)))


if __name__ == "__main__":
    unittest.main()
