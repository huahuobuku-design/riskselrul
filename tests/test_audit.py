"""Regression for archived underscore column names and all-rejected cells."""
import json
import numpy as np
import pytest

pd = pytest.importorskip('pandas', reason='independent table audit is a training extra')
from riskselrul.audit import audit_run


def test_saved_flags_empty_retention_and_corruption_detection(tmp_path):
    heads = ('H_log','H_sq','pred_only','fault_indicator')
    predictor = 'ridge_clean'
    tables, metric_rows = [], []
    for scenario in ('clean','A_bias+2'):
        # Every error is exactly -1, making the expected MSE exactly one for any k.
        table = pd.DataFrame(dict(engine=[f'test:{i}' for i in range(1,101)], scen=scenario,
                                  y_raw=np.ones(100), pred_ridge_clean=np.zeros(100)))
        for head in heads:
            score = np.arange(100)+(0 if scenario=='clean' else 100)
            table[f'score_{predictor}_{head}'] = score
            table[f'acc_{predictor}_{head}'] = (score<=79.2).astype(int)
            table[f'sel80_{predictor}_{head}'] = (np.arange(100)<80).astype(int)
            row = dict(ds='FD001',seed=42,stream=9001,pred=predictor,head=head,scen=scenario)
            row.update({'full.MSE':1.0,'thr.thr':79.2,'thr.k':80 if scenario=='clean' else 0,
                        'thr.ret_MSE':1.0 if scenario=='clean' else np.nan})
            for k in (20,40,60,80,100):
                row[f'k{k}.sel.MSE']=1.0
                row[f'k{k}.oracle.MSE']=1.0
            metric_rows.append(row)
        tables.append(table)
    path = tmp_path/'test_FD001_s42_st9001.csv'
    combined = pd.concat(tables)
    combined.to_csv(path,index=False)
    calibration = pd.DataFrame({f'score_{predictor}_{head}':np.arange(100) for head in heads})
    calibration.to_csv(tmp_path/'calib_FD001_s42.csv',index=False)
    pd.DataFrame(metric_rows).to_csv(tmp_path/'metrics_long.csv',index=False)
    (tmp_path/'thresholds_FD001_s42.json').write_text(json.dumps({predictor:{head:79.2 for head in heads}}))
    result, direction = audit_run(tmp_path)
    assert result['checked'] > 100 and result['mismatches']==0
    assert all(row['denominator']==0 and row['direction_agreement'] is None for row in direction)
    # Corrupt a saved acceptance flag; the auditor must catch it independently.
    combined.iloc[0, combined.columns.get_loc('acc_ridge_clean_H_log')]=0
    combined.to_csv(path,index=False)
    corrupted, _ = audit_run(tmp_path)
    assert corrupted['mismatches']==1
