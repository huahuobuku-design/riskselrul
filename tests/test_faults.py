"""Fault generation must be parseable, deterministic and physically sensible."""

import unittest

import numpy as np

from riskselrul.faults import rng as R
from riskselrul.faults import scenarios as F


class TestFaults(unittest.TestCase):
    def setUp(self):
        self.raw = np.arange(30 * 14, dtype=float).reshape(30, 14) + 1.0
        self.SD = np.ones(14)

    def test_parse_labels(self):
        self.assertIsNone(F.parse("clean"))
        self.assertEqual(F.parse("A_dropout1"), ("A", "dropout1", "dropout", 1, 0))
        self.assertEqual(F.parse("B_stuck3"), ("B", "stuck3", "stuck", 1, 0))
        self.assertEqual(F.parse("A_bias+4"), ("A", "bias4", "bias", 1, 4))
        self.assertEqual(F.parse("B_bias-2"), ("B", "bias2", "bias", -1, 2))
        self.assertEqual(F.parse("A_drift+4"), ("A", "drift4", "drift", 1, 4))

    def test_scenario_lists(self):
        self.assertEqual(F.TEST_SCEN[0], "clean")
        self.assertEqual(len(F.TEST_SCEN), 21)
        self.assertEqual(len(F.TRAIN_SCEN), 8)

    def test_draw_is_deterministic(self):
        a = F.draw("FD001", 42, "final", 9001, "test:1", 100, "bias4")
        b = F.draw("FD001", 42, "final", 9001, "test:1", 100, "bias4")
        self.assertEqual(a, b)
        on, pos, _k = a
        self.assertTrue(5 <= on <= 25)
        self.assertEqual(len(pos), 3)

    def test_dropout_makes_nan_after_onset(self):
        x = F.apply(self.raw, "A_dropout1", 10, [0], self.SD)
        j = F.SENS.index(F.A[0])
        self.assertTrue(np.all(np.isnan(x[10:, j])))
        self.assertFalse(np.any(np.isnan(x[:10])))

    def test_stuck_freezes_channel(self):
        x = F.apply(self.raw, "A_stuck1", 10, [0], self.SD)
        j = F.SENS.index(F.A[0])
        self.assertTrue(np.all(x[10:, j] == x[10, j]))

    def test_bias_amplitude_is_in_sd_units(self):
        SD = np.full(14, 2.0)
        x = F.apply(self.raw, "A_bias+4", 10, [0], SD)
        j = F.SENS.index(F.A[0])
        self.assertTrue(np.allclose(x[10:, j] - self.raw[10:, j], 8.0))

    def test_impute_uses_channel_mean(self):
        x = self.raw.copy()
        x[5:, 3] = np.nan
        y = F.impute(x, np.arange(14, dtype=float))
        self.assertTrue(np.all(y[5:, 3] == 3.0))

    def test_engine_order_key_is_stable_sha256(self):
        self.assertEqual(R.engine_order_key("test:1"), R.engine_order_key("test:1"))
        self.assertEqual(len(R.engine_order_key("test:1")), 64)


if __name__ == "__main__":
    unittest.main()
