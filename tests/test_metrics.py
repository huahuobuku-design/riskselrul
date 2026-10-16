import numpy as np
import pytest
from riskselrul.metrics import calibrate_threshold, fixed_k_metrics, prediction_metrics, risk_coverage_curve, stable_risk_order


def test_zero_retention_is_undefined():
    result = fixed_k_metrics([10, 20], [12, 18], [0, 1], 0)
    assert result['retained']['mse'] is None
    assert result['random_expected_mse'] is None
    assert prediction_metrics([], [])['rmse'] is None


def test_order_never_depends_on_target_and_ties_follow_ids():
    ids = ['test:1', 'test:2', 'test:3']
    a = fixed_k_metrics([1, 2, 3], [0, 1, 8], [1, 1, 1], 2, ids)
    b = fixed_k_metrics([1, 2, 3], [99, 4, -3], [1, 1, 1], 2, ids)
    assert a['selected_indices'] == b['selected_indices']
    permutation = np.array([2, 0, 1])
    before = np.array(ids)[stable_risk_order([1, 1, 1], ids)]
    after = np.array(ids)[permutation][stable_risk_order([1, 1, 1], np.array(ids)[permutation])]
    assert list(before) == list(after)


def test_oracle_bound_full_retention_and_random_reference():
    rng = np.random.default_rng(9)
    p, y, s = rng.normal(size=(3, 30))
    curve = risk_coverage_curve(p, y, s)
    assert all(r['retained_mse']+1e-12 >= r['oracle_mse'] for r in curve)
    assert curve[-1]['retained_mse'] == pytest.approx(np.mean((p-y)**2))
    assert curve[-1]['oracle_mse'] == pytest.approx(curve[-1]['retained_mse'])
    assert len({r['random_expected_mse'] for r in curve}) == 1
    # Exhaustively enumerate size-one random retention to check the MSE expectation.
    assert np.mean([(p[i]-y[i])**2 for i in range(30)]) == pytest.approx(curve[0]['random_expected_mse'])


@pytest.mark.parametrize('k', [-1, 3, 1.5, True])
def test_invalid_retained_count(k):
    with pytest.raises(ValueError):
        fixed_k_metrics([1, 2], [1, 2], [0, 1], k)


def test_quantile_targets_do_not_guarantee_acceptance_under_shift():
    threshold = calibrate_threshold(np.arange(100), .8)
    assert np.mean(np.arange(100)+100 <= threshold) == 0
    with pytest.raises(ValueError):
        calibrate_threshold([], .8)


def test_nonfinite_inputs_rejected():
    with pytest.raises(ValueError):
        prediction_metrics([np.inf], [0])
