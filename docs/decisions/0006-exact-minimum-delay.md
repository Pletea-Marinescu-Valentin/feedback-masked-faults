# 0006. Exact minimum alarm delay of the kappa-mixture e-detector

- Status: accepted
- Date: 2026-10-04

## Context

The minimum delay D_min of the kappa-mixture e-CUSUM was computed as
ceil(ln(K / alpha) / max_kappa ln c_kappa), with c_kappa = kappa p_min^(kappa - 1)
the largest possible e-value. That is the time at which the best kappa alone
reaches K / alpha, which is sufficient for an alarm but not necessary: the
other kappas also add to the average. The paper states D_min as a time before
which no alarm is possible, so the old value can be too large. On a grid of
close kappas (0.30 to 0.38, n = 450, alpha = 1e-3) it gives 11 instead of 9.

## Decision

D_min is the exact fastest alarm of the mixture,

    D_min = min{ t : (1 / K) sum_kappa max(c_kappa, c_kappa^t) >= 1 / alpha },

because each e-CUSUM grows with every e-value and equals max(c, c^t) when all
e-values equal c. Scores above every calibration score attain it. It lies
between ln(1 / alpha) / max ln c_kappa and the old value. Implemented in
`fmf.theory.delay.e_detector_min_delay`; the test simulates the attaining
scores on a spread and a close kappa grid.

## Consequences

Only reported minimum delays change; detector statistics, alarms and delays
do not. With the configured grid (0.35, 0.6, 0.85) the best kappa dominates
and most values are unchanged. Changed: Table 1, 15-min e-detector with DKW,
2.5 h to 2.2 h (cooling valve); Fig. 4, dotted D_min at ARL0 of 365 and 3000
days, 20 h to 19 h and 24 h to 23 h; several values in
`results/tables/tab_far_ers.csv` not quoted in the paper. The design numbers
of the discussion (`tab_design`) are unchanged.
