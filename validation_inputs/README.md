# Two distinct profiles under review

The adjacent riskselrul-0.1.0.tar.gz is the byte-identical local alpha source
distribution prepared on 2026-10-17. SHA-256:
86f8d015f02dccdc90bc0a439b87b1a6c377153208288e9b6a008b45cbf56117.

It has a different module layout, API and test suite from this repository's
1.0.0 snapshot. It is included for isolated CI verification, not as a claim that
the two versions are equivalent. Each matrix job installs exactly one profile.
The alpha uses pytest; the repository 1.0.0 tests use unittest. No raw NASA data,
credentials, paper PDFs or private chats are included. CI synthetic state checks
do not measure model accuracy. The four existing raw-data state tests may skip
when NASA inputs are unavailable; all 78 states are also exercised with synthetic
numeric fixtures in a separate test.
