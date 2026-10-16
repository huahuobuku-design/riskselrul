"""Example 3 - load a released lean state and run inference.

No training, no scikit-learn, no pandas: only NumPy and the shipped state.

    python examples/03_load_lean_state.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from riskselrul.deploy import lean  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "..", "deploy", "states", "FD003_seed42_all_linear_57.npz")


def main():
    state = lean.load(STATE)
    print("state keys:", ", ".join(sorted(state)))

    # Raw windows must have shape (n, 30, 21) in original sensor units; NaN is allowed.
    rng = np.random.default_rng(0)
    raw = 100.0 + 5.0 * rng.normal(size=(5, 30, 21))
    raw[0, 20:, 4] = np.nan  # simulate a sensor dropout

    pred, risk, accept = lean.decide(raw, state, condition="C2", coverage=0.8)
    for i in range(len(pred)):
        print("window %d: pred=%8.3f  risk=% .6f  accept=%s"
              % (i, pred[i], risk[i], bool(accept[i])))

    print("accepted %d / %d" % (int(accept.sum()), len(accept)))


if __name__ == "__main__":
    main()
