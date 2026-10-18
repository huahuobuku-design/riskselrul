# Release and submission status

Updated 2026-10-18. The canonical SoftwareX candidate is **0.1.1.dev0 on softwarex-dev**. Clone this branch explicitly. The main branch retains a different historical 1.0.0 implementation; validated-v0.1.0 is the frozen computational baseline. Old branches and tags remain intact.

## Completed records

- Author: Jinxian Wu. Affiliation: College of Publishing, University of Shanghai for Science and Technology, Shanghai 200093, China. Contact: 2435050925@st.usst.edu.cn. The author reports no competing interests.
- Apache-2.0 license is included. NASA data and external packages remain separate works; raw NASA files are not redistributed.
- Candidate installed-wheel CI passed 21 tests on Ubuntu and Windows/Python 3.12; see reproducibility/candidate_ci_verification.json and ci_test_counts.json.
- The frozen expansion includes six dataset/seed fits, eighteen fault-stream cases and independent local arithmetic/preprocessing/interval checks in S2. Received execution logs and local model refitting have distinct evidence scopes.
- Manuscript v0.6, API reference, minimal example and supplements are public.

## This finalization pass

- Add self-designed functional user tasks, distinct from external human adoption.
- Extend fresh-runner testing to Python 3.10 and 3.12 on Ubuntu and Windows.
- Add fresh-download, strict-hash, full training and audit jobs for FD001 and FD003, seed 42 / stream 9001 / 2,000 bootstrap draws. Record execution evidence separately after completion; configuration alone is not PASS.
- Repeat the same raw-data path locally after the clean-runner check.

## Author and publication decisions

- Funding: —. ORCID: —. External human adoption: none documented.
- Formal candidate release, immutable archival deposit and DOI: on hold.
- Final author approval, current submission-form fields, any publication fee decision and actual submission: not yet performed.

The goal is a useful software companion article and practical submission experience. Institutional assessment and acceptance are not completion criteria. Missing external adoption does not block preparation. This pass ends after its defined checks, documentation and author report; it does not automatically expand into a new algorithm study.
