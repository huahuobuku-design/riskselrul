import numpy as np
from riskselrul._legacy.faults import SENS, apply, draw, parse


def test_fault_pairs_share_onset_and_relative_channel_positions():
    a = draw('FD001', 42, 'final', 9001, 'test:8', 50, parse('A_bias+4')[1])
    b = draw('FD001', 42, 'final', 9001, 'test:8', 50, parse('B_bias-4')[1])
    assert a == b
    raw = np.arange(30*14, dtype=float).reshape(30, 14)
    before = raw.copy()
    pos = apply(raw, 'A_bias+4', a[0], a[1], np.ones(14))
    neg = apply(raw, 'A_bias-4', a[0], a[1], np.ones(14))
    np.testing.assert_array_equal((pos+neg)/2, raw)
    np.testing.assert_array_equal(raw, before)
