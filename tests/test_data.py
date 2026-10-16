import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
from riskselrul.data import extract_cmapss, inspect_data


def _write_minimal(root):
    rows = np.array([[engine, cycle, *np.zeros(24)] for engine in range(1, 101) for cycle in (1, 2)])
    for prefix in ('train', 'test'):
        np.savetxt(root/f'{prefix}_FD001.txt', rows)
    np.savetxt(root/'RUL_FD001.txt', np.arange(100))


def test_shape_and_hash_policy(tmp_path):
    _write_minimal(tmp_path)
    result = inspect_data(tmp_path)
    assert result['files'][0]['shape'] == [200, 26]
    assert not result['files'][0]['matches_archived_bytes']
    with pytest.raises(ValueError, match='archived NASA bytes'):
        inspect_data(tmp_path, require_known_hashes=True)


def test_duplicate_cycle_fails(tmp_path):
    _write_minimal(tmp_path)
    path = tmp_path/'train_FD001.txt'
    rows = np.loadtxt(path)
    rows[1, 1] = 1
    np.savetxt(path, rows)
    with pytest.raises(ValueError, match='Nonconsecutive'):
        inspect_data(tmp_path)


def test_zip_traversal_rejected(tmp_path):
    archive = tmp_path/'bad.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('../escape.txt', 'bad')
    with pytest.raises(ValueError, match='Unsafe'):
        extract_cmapss(archive, tmp_path/'target')
    assert not (tmp_path/'escape.txt').exists()


def test_zip_duplicate_rejected(tmp_path):
    archive = tmp_path/'duplicate.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('one/train_FD001.txt', '1')
        stream.writestr('two/train_FD001.txt', '1')
    with pytest.raises(ValueError, match='Duplicate'):
        extract_cmapss(archive, tmp_path/'target')
