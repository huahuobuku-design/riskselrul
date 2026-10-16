"""Metric identities that must hold on any data."""

import unittest

import numpy as np

from riskselrul import config
from riskselrul.selective.decision import cell, order
from riskselrul.selective.metrics import metrics


class TestMetricIdentities(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 100
        self.p = rng.uniform(0, 125, self.n)
        self.t = rng.uniform(0, 125, self.n)
        self.s = rng.uniform(0, 1, self.n)
        self.engines = [f"test:{u}" for u in range(1, self.n + 1)]
        self.draws = np.zeros(self.n)

    def test_full_mse_definition(self):
        m = metrics(self.p, self.t)
        self.assertAlmostEqual(m["MSE"], float(np.mean((self.p - self.t) ** 2)), places=12)
        self.assertAlmostEqual(m["RMSE"], float(np.sqrt(m["MSE"])), places=12)

    def test_k100_selection_equals_full_sample(self):
        thr = float(np.quantile(self.s, 0.8))
        c = cell(self.p, self.t, self.s, thr, self.engines, self.draws)
        self.assertAlmostEqual(c["k100"]["sel"]["MSE"], c["full"]["MSE"], places=12)
        self.assertAlmostEqual(c["k100"]["oracle"]["MSE"], c["full"]["MSE"], places=12)

    def test_oracle_is_a_lower_bound(self):
        c = cell(self.p, self.t, self.s, 0.5, self.engines, self.draws)
        for k in config.FIXED_K:
            self.assertLessEqual(c[f"k{k}"]["oracle"]["MSE"], c["full"]["MSE"] + 1e-9)

    def test_order_is_label_free_and_ascending(self):
        o1 = order(self.s, self.engines, self.draws)
        o2 = order(self.s, self.engines, self.draws)
        self.assertTrue(np.array_equal(o1, o2))
        self.assertTrue(np.all(np.diff(self.s[o1]) >= -1e-12))

    def test_threshold_acceptance_matches_risk(self):
        thr = float(np.quantile(self.s, 0.8))
        c = cell(self.p, self.t, self.s, thr, self.engines, self.draws)
        self.assertEqual(c["thr"]["k"], int((self.s <= thr).sum()))


if __name__ == "__main__":
    unittest.main()
