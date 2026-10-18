# Software article positioning

Working title: **RiskSelRUL: reproducible sensor-fault experiments and independent diagnostics for selective remaining useful life prediction**.

The software contribution is a reusable workflow that joins data validation, a versioned grouped-OOF protocol, transparent acceptance decisions, and independent numerical auditing. It also separates offline ranking diagnostics from frozen-threshold inference. These claims are supported by executable interfaces and validation records, rather than an assertion that residual prediction or selective regression is new.

An illustrative example should start from NASA files, produce per-engine predictions, reconstruct a risk-coverage curve at fixed retained counts, then independently verify saved decisions. Archived numeric inference states provide a second, explicitly distinct example of deployment-format portability. The article must not imply that the raw-data training example regenerated those older states.

Potential research reuse includes replacing a point predictor, comparing selectors at fixed predictions and retained counts, examining changes under unseen sensor groups, and diagnosing acceptance collapse without counting rejection as zero error. These are intended uses, not evidence that independent researchers have already adopted the software.

The current companion manuscript is paper/manuscript_v0.6.md and follows SoftwareX OSP template version 6. It describes software interfaces, executed examples, validation and reuse. Author and contact are recorded in docs/release_checklist.md. Final release and DOI archival are on hold. Self-designed user tasks in docs/user_tasks.md demonstrate practical workflows without implying external adoption.

Existing-work anchors (not a priority or superiority claim):

- SelectiveNet, ICML 2019: https://proceedings.mlr.press/v97/geifman19a.html
- Regression with reject option and application to kNN, NeurIPS 2020: https://proceedings.neurips.cc/paper/2020/hash/e8219d4c93f6c55c6b10fe6bfe997c6c-Abstract.html
- Conformal Prediction Intervals for Remaining Useful Lifetime Estimation, 2023: https://papers.phmsociety.org/index.php/ijphm/article/view/3417
- Conformal Regression with Reject Option, 2024: https://proceedings.mlr.press/v230/johansson24a.html
- RobustUQ, 2025: https://doi.org/10.1016/j.ress.2025.111177
- MSCCAN, 2025: https://doi.org/10.1109/TIM.2025.3551011

The direct RUL literature contains significant overlap. Full-method comparison for RobustUQ/MSCCAN remains incomplete. The software article should explain its practical contribution and boundaries without turning unread papers into claimed gaps.
