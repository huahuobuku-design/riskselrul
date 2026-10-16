"""A NumPy-only input/output example for the validated 0.1.0 API.

All values are synthetic user-supplied predictions/scores, not NASA performance.
Run: python quickstart_arrays.py --out NEW_DIRECTORY
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from riskselrul import fixed_k_metrics
from riskselrul.metrics import calibrate_threshold, prediction_metrics, risk_coverage_curve


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    prediction=np.array([10,12,9,15,8,13,11,14],dtype=float)
    target=np.array([11,10,8,14,10,12,10,11],dtype=float)
    # These scores are supplied independently, before evaluation labels.
    score=np.array([.7,.1,.4,.8,.3,.2,.6,.5])
    ids=[f'example-{i+1}' for i in range(8)]
    fixed=fixed_k_metrics(prediction,target,score,k=4,sample_ids=ids)
    threshold=calibrate_threshold([.1,.2,.3,.4,.5],acceptance_rate=.6)
    accepted=score<=threshold
    threshold_metrics=prediction_metrics(prediction[accepted],target[accepted])
    empty=fixed_k_metrics(prediction,target,score,k=0,sample_ids=ids)
    full=fixed_k_metrics(prediction,target,score,k=8,sample_ids=ids)
    changed=fixed_k_metrics(prediction,target+20,score,k=4,sample_ids=ids)
    manual_order=np.argsort(score,kind='stable')
    assert fixed['selected_indices']==manual_order[:4].tolist()
    assert fixed['selected_indices']==changed['selected_indices']
    assert np.isclose(fixed['retained']['mse'],2.5)
    assert np.isclose(full['retained']['mse'],2.75)
    assert np.isclose(threshold_metrics['mse'],3.)
    assert threshold_metrics['n']==3
    assert empty['retained']['mse'] is None
    result=dict(kind='synthetic API example; not a benchmark',n=8,
                fixed_k=dict(k=4,acceptance_rate=fixed['acceptance_rate'],mse=fixed['retained']['mse'],
                             selected_ids=[ids[i] for i in fixed['selected_indices']],oracle_mse=fixed['oracle']['mse'],
                             random_expected_mse=fixed['random_expected_mse']),
                threshold=dict(calibration_target=.6,value=threshold,n=threshold_metrics['n'],acceptance_rate=float(accepted.mean()),mse=threshold_metrics['mse']),
                zero_retained_mse=empty['retained']['mse'],all_retained_mse=full['retained']['mse'],
                independent_numeric_checks='PASS')
    with (out/'input.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['id','prediction','target','risk_score'])
        w.writerows(zip(ids,prediction,target,score))
    curves=risk_coverage_curve(prediction,target,score,ids)
    with (out/'curve.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
