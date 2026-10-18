"""Self-designed user tasks; synthetic inputs, not external adoption evidence."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from riskselrul import calibrate_threshold, fixed_k_metrics, prediction_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=False)
    prediction = [1., 2., 3., 4.]
    target = [1., 0., 2., 0.]
    ids = ['a', 'b', 'c', 'd']
    # Scores are supplied as inputs, before outcomes are evaluated.
    score_a = [.1, .3, .2, .4]
    score_b = [.4, .1, .3, .2]
    with (out / 'input.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'prediction', 'target', 'score_a', 'score_b'])
        writer.writerows(zip(ids, prediction, target, score_a, score_b))
    # Task 1: compare selectors while holding predictor, cohort and k fixed.
    a = fixed_k_metrics(prediction, target, score_a, 2, ids)
    b = fixed_k_metrics(prediction, target, score_b, 2, ids)
    assert a['selected_indices'] == [0, 2] and a['retained']['mse'] == .5
    assert b['selected_indices'] == [1, 3] and b['retained']['mse'] == 10.
    # Task 2: use one frozen calibration threshold on a shifted cohort.
    threshold = calibrate_threshold([.1, .2, .3, .4], .5)
    shifted_scores = np.array([.6, .7, .8, .9])
    accepted = shifted_scores <= threshold
    shifted = prediction_metrics(np.asarray(prediction)[accepted], np.asarray(target)[accepted])
    assert shifted['n'] == 0 and shifted['mse'] is None
    # Task 3: demonstrate that fixed-count ranking is a different question.
    diagnostic = fixed_k_metrics(prediction, target, shifted_scores, 2, ids)
    assert diagnostic['retained']['n'] == 2 and diagnostic['retained']['mse'] == 2.
    result = dict(
        evidence='Self-designed functional scenarios using synthetic arrays; not external user adoption',
        same_k_comparison=dict(k=2, a_mse=.5, b_mse=10., full_mse=a['full']['mse']),
        frozen_threshold=dict(value=threshold, calibration_target=.5, test_acceptance=0., metrics=shifted),
        offline_fixed_k=dict(acceptance=diagnostic['acceptance_rate'], mse=diagnostic['retained']['mse']),
        status='PASS')
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
