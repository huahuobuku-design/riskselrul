from pathlib import Path
import numpy as np
import pytest
from riskselrul.deploy import available_states, decide, load_state, predict


def test_all_frozen_states_match_archived_forward():
    # Goldens were generated with the sealed WP04A implementation, not this module.
    with np.load(Path(__file__).parent/'fixtures/legacy_forward.npz', allow_pickle=False) as golden:
        assert len(available_states()) == 78
        assert set(available_states()) == set(golden['names'])
        for i, name in enumerate(golden['names']):
            state = load_state(str(name))
            p, risk, accepted = decide(golden['raw'], state, str(golden['conditions'][i]), .8)
            np.testing.assert_allclose(p, golden['predictions'][i], rtol=1e-11, atol=1e-9)
            np.testing.assert_allclose(risk, golden['risks'][i], rtol=1e-11, atol=1e-9)
            np.testing.assert_array_equal(accepted, golden['accepted'][i])


def test_input_contract_and_unsupported_threshold():
    state = load_state('FD001_seed42_all_linear_57.npz')
    with pytest.raises(ValueError, match='shape'):
        predict(np.zeros((2, 21)), state)
    with pytest.raises(ValueError):
        predict(np.full((1, 30, 21), np.inf), state)
    with pytest.raises(ValueError, match='threshold'):
        decide(np.zeros((1, 30, 21)), state, 'C3_unknown', .8)
    p, risk = predict(np.empty((0, 30, 21)), state)
    assert p.shape == risk.shape == (0,)


def test_state_objects_not_deserialized(tmp_path):
    state_path = tmp_path/'object.npz'
    np.savez(state_path, malicious=np.array([{'unsafe': True}], dtype=object))
    with pytest.raises(ValueError):
        load_state(state_path)
