# 0002. Detector design for C1 and C2

- Status: accepted
- Date: 2026-10-03

## Context

Command residuals are autocorrelated; at 1-min resolution the per-sample
information is small, the residual is heavy-tailed on real data, and the
conformal e-detector has an e-value cap set by the calibration size.

## Decision

- C1 (theory checks on the synthetic loop): AR(8) model fitted on fault-free
  calibration residuals, CUSUM on the whitened innovations tuned to the
  quasi-static shift. Delays predicted with the Brownian-limit law, not
  ln(ARL0)/I.
- C2 (calibrated detector): scores are means of the command residual over
  non-overlapping blocks within a day, standardized by the spread of the
  calibration blocks; one-sided conformal p-values against the calibration
  blocks; DKW correction with delta = 0.05; power calibrators averaged over
  kappa in {0.35, 0.6, 0.85} (e-CUSUM, unfloored statistics); two one-sided
  detectors at alpha / 2. Block length 60 min, the shortest block on which the
  false-alarm rate held on the real ERS test days for all three actuators
  (`experiments/tab_far_ers.py`).
- Minute-level kappa grid {0.8, 0.9, 0.95, 0.98, 0.99}, kept only for the
  comparison in Table 1.

## Consequences

- The guarantee needs independent scores; 60-min blocks are an empirical
  choice for the ERS data and must be rechecked on other buildings.
- Minimum delay (Proposition 4) of 13-15 h with 19-25 calibration days at
  ARL0 = 30 days; longer calibration shortens it (`experiments/tab_design.py`).
- The Gaussian CUSUM is reported as a reference only; its false-alarm rate is
  not met on real data.
