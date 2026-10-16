"""Numerical demonstration only: these are synthetic errors, not NASA results."""
from riskselrul import risk_coverage_curve

for row in risk_coverage_curve([30, 22, 20, 4], [25, 20, 10, 5], [2, 1, 3, 0]):
    print(row)
