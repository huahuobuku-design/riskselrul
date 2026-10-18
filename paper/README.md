# SoftwareX manuscript and supplementary data

The manuscript v0.6 is the current working draft. It cites the fixed public source commit and verified candidate CI results. It has not been submitted.
The current development source is on `softwarex-dev`; the frozen experiment baseline
is tagged `validated-v0.1.0`. No new release or DOI archive is created by this update.

- [Manuscript](manuscript_v0.6.md) and [editable Word file](RiskSelRUL_SoftwareX_v0.6.docx).
- [Supplement S1](supplementary/Supplementary_S1_v0.3.zip): earlier selected-run verification.
- [Supplement S2](supplementary/Supplementary_S2_v0.5.zip): full expansion outputs, logs and portable checks.
- [File hashes](supplementary/SHA256.json).
- [Chinese verification report](../reproducibility/SX_EXPAND_review_zh.md).

Extract S2 and follow its README to verify hashes or recompute metrics/intervals.
Original NASA files are not included. Rebuild the numerical figure with NumPy,
pandas and Matplotlib using `python paper/figure_from_s2.py --s2 EXTRACTED_S2 --out NEW_OUTPUT`.
