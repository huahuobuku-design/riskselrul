"""Default configuration mirroring the frozen experiment protocol.

These constants document the released protocol. Only a subset is exposed as CLI
flags; everything else is intentionally fixed so that released results stay
reproducible.
"""

# Label handling
CAP = 125  # training label = min(RUL, CAP); evaluation uses the raw uncapped truth

# Data
DATASETS = ["FD001", "FD003"]
TRAIN_SEEDS = [42, 43, 44]
WINDOW = 30
N_SENSORS = 14
N_FEATURES = 56

# Splits / streams (RNG stream identifiers, not random seeds)
TEST_STREAMS = [9001, 9002, 9003]
AUG_STREAM = 2026101601
OOF_STREAM = 2026101602
CALIB_STREAM = 2026101603

# Out-of-fold construction
OOF_FOLDS = 5
OOF_TRUNCATIONS = 10

# Selective evaluation
FIXED_K = [20, 40, 60, 80, 100]
MAIN_K = 80
COVERAGE = 0.8
ERROR_BAND = 20
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261016
