"""Exercise the archived inference API with SYNTHETIC windows, not an accuracy benchmark."""
import numpy as np
from riskselrul.deploy import load_state, decide

state = load_state('FD001_seed42_all_linear_57.npz')
windows = np.random.default_rng(0).normal(state['raw_MU21'], 1.0, size=(2, 30, 21))
windows[1, 20:, 1] = np.nan
prediction, risk, accepted = decide(windows, state, 'C2', .8)
print('SYNTHETIC API EXAMPLE; not a performance claim')
for values in zip(prediction, risk, accepted):
    print(values)
