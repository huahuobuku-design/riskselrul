# Limitations

This software standardizes reproducible experiments and diagnostic accounting. It does not establish a new algorithm or a general risk guarantee.

- The two C-MAPSS subsets have been reused during exploratory project development. Multiple seeds are not independent external datasets or prospective confirmation.
- Injected faults are simplified simulation perturbations, not a complete physical fault model. Augmentation on one channel group may worsen performance on another group.
- A learned residual score does not reliably identify whether each individual fault increased the prediction error. Static risk ranking and paired fault-damage recognition are distinct tasks.
- Severe sensor bias can collapse the base predictor. Even label-informed oracle retention can fail to recover clean-input error at useful retained counts.
- Training-label clipping changes the underlying predictor and error scale. A larger relative benefit against a predictor's own random baseline can accompany worse absolute retained error. Compare fixed predictors and equal retained counts.
- Frozen-threshold acceptance can fall far below the calibration target under shift. A rejected prediction has no retained error; empty retained sets are NA/None.
- The 57/99 feature WP04A states are computational/accuracy trade-offs. Historical FD003 comparisons showed retained-error costs. Feature dimensions are not physical sensor counts. Stored array bytes are not peak process memory or energy.
- Fixed-k and oracle rankings use the test cohort and are offline diagnostics. They are not online threshold calibration or a deployable policy.
- The historical WP05A anchor percentage omitted zero risk changes; use the new shared-denominator audit output for direction comparisons.
- A migrated training reproduction is not a reproduction of every historical work package. Legacy inference state parity is not an independent validation of all upstream training caches.
- The historical project recorded failed verification attempts and a later fixed-k arithmetic correction. Some early failure logs were incomplete. This package preserves those limitations in provenance; a successful current run cannot recreate missing history.
- Key close literature includes RobustUQ and MSCCAN, whose complete methods were not obtained in the project's review. No claim that their methods lack particular components is made.
- A model output or accepted prediction is not a maintenance action. No fleet decision policy or real-device safety performance is validated.
