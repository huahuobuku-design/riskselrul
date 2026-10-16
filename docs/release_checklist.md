# Release and submission status

This is the separately tagged validated 0.1.0 profile, not the repository's main-branch 1.0.0 implementation. Runtime source, numeric states and frozen configuration match the reviewed 0.1.0 inventory; documentation, a quickstart example and CI files were added for publication.

- Apache-2.0 license text is included under LICENSE, LICENSE.txt and Licence.txt. Top-level dependencies are separately installed, not relicensed or vendored. Their installed license metadata is recorded in reproducibility/license_inventory.json; this is not an audit of all binary/transitive redistribution obligations.
- Ubuntu/Windows Python 3.12 installation and 18 tests passed for the exact frozen 0.1.0 source archive. See reproducibility/ci_verification.json. CI does not rerun raw-data training.
- An AI-operated Debian/Python 3.13.5 raw-data run returned FD001/seed42/stream9001 results; independently recomputed tables passed 5,392 checks. This is not independent human adoption. See reproducibility/grok_numeric_review.json.
- Full two-dataset/three-seed/three-stream retraining has not been repeated for this release.
- Human author names, affiliation, support contact, contributor/funding/conflict declarations and final attribution review: —.
- Persistent software/data archive DOI: —. A GitHub release alone is not an immutable archival deposit.
- Article author approval and journal submission: —.

Official guide/template and any current submission requirements must be checked before submission. The accompanying draft follows SoftwareX OSP template version 6 (March 2026); no acceptance or submission is claimed.
