# Release and submission status

Updated 2026-10-18. The canonical SoftwareX candidate is **0.1.1.dev0 on softwarex-dev**. Clone this branch explicitly. The main branch retains a different historical 1.0.0 implementation; validated-v0.1.0 is the frozen computational baseline. Old branches and tags remain intact.

## Completed records

- Author: Jinxian Wu. Affiliation: College of Publishing, University of Shanghai for Science and Technology, Shanghai 200093, China. Contact: 2435050925@st.usst.edu.cn. The author reports no competing interests.
- Apache-2.0 license is included. NASA data and external packages remain separate works; raw NASA files are not redistributed.
- Candidate installed-wheel CI passed 21 tests on Ubuntu and Windows/Python 3.12; see reproducibility/candidate_ci_verification.json and ci_test_counts.json.
- The frozen expansion includes six dataset/seed fits, eighteen fault-stream cases and independent local arithmetic/preprocessing/interval checks in S2. Received execution logs and local model refitting have distinct evidence scopes.
- Manuscript v0.7, API reference, minimal example and supplements are public.

## Finalization checks completed

- Three self-designed functional user tasks passed; these are not external adoption.
- Four Ubuntu/Windows and Python 3.10/3.12 combinations each passed 21 tests.
- Two fresh Ubuntu raw jobs and the same two local Windows raw workflows passed.
- Actual output files and selected arithmetic/fold boundaries were independently checked.
- Cross-platform refit differences are documented in docs/cold_start_validation.md and S3.
- Manuscript v0.7 is the current draft. The previous v0.6 and old tags remain preserved.

## Author and publication decisions

- Funding: —. ORCID: —. External human adoption: none documented.
- Formal candidate release, immutable archival deposit and DOI: on hold.
- Final author approval, current submission-form fields, any publication fee decision and actual submission: not yet performed.

The goal is a useful software companion article and practical submission experience. Institutional assessment and acceptance are not completion criteria. Missing external adoption does not block preparation. This pass ends after its defined checks, documentation and author report; it does not automatically expand into a new algorithm study.
