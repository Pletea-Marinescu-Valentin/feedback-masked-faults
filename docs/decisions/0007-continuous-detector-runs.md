# 0007. Detectors run continuously through the test days

- Status: accepted
- Date: 2026-10-20

## Context

The false-alarm study of 0003 restarted every detector at the start of each test
day. The hourly e-detector with DKW correction cannot alarm before its minimum
delay (0006): 15, 36 and 13 h for the cooling valve, heating valve and supply fan
at ARL0 = 30 days, against about 14, 11 and 13 hourly blocks per test day. Its
zero false-alarm rate in Table 1 therefore held by construction rather than by
calibration.

The control runs of the faulted-day study had the same limit, with minimum delays
of 10, 20 and 9 h on single days, and each control day was part of its own
calibration set.

Separately, the negative-sign detectors of the false-alarm study compared negated
test scores with unnegated calibration scores. That is valid only for symmetric
scores, and the whitened residuals are heavy-tailed and skewed.

## Decision

- **False-alarm study.** Each detector runs through the test days in
  chronological order and restarts only after an alarm. Summed over restarts,
  the averaged Shiryaev-Roberts statistics minus the number of observations form
  a supermartingale. The expected number of false alarms in T observations is
  therefore at most alpha T, and the nominal rate per operating day stays
  1 / ARL0.
- **Signs.** Both one-sided detectors use calibration scores signed like their
  test scores.
- **Faulted-day study.** The faulted unit is unchanged: each fault lasts one day,
  so each faulted day is one run. The control monitors the fault-free unit with
  detectors calibrated on the other chronological half of its days, within
  seasons and in both directions. The stream runs through the days in order and
  restarts only after an alarm.

## Consequences

**Table 1 at ARL0 = 30 days, in false alarms per day (cooling / heating / fan).**

| Detector | Before | After |
|---|---|---|
| e-detector with DKW, 60 min | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| e-detector with DKW, 30 min | 0.06 / 0.00 / 0.08 | 0.11 / 0.17 / 0.12 |
| e-detector without DKW, 60 min | 0.33 / 0.17 / 0.25 | 0.39 / 0.50 / 0.33 |
| Gaussian CUSUM, 1 min | 0.89 / 0.17 / 0.29 | 0.89 / 0.33 / 0.29 |

The Gaussian CUSUM on 60-min blocks is unchanged at 1.28 / 0.33 / 0.79. The
conclusion stands: the DKW e-detector meets the nominal rate on all three
actuators only on blocks of 60 minutes or more, and the other detectors do not.

**Limits of the test.** The heating valve has 6 test days against a minimum
delay of 36 h, so its 60-min column remains a weak test.

**Control count.** "0 of 196 runs" becomes 1 alarm in 196 monitor-days: one
alarm on the cooling valve in 37 days, none on the other five monitors. The
nominal rate allows at most 6.5 in expectation. With half the days as
calibration, the control minimum delays are 14, 36 and 12 h.

**Numbers and outputs.** The macro TabDetectionErsControlRuns is replaced by
TabDetectionErsControlDays. The control counts are written to
`results/tables/tab_detection_ers_control.csv`. The faulted-day results are
unchanged.
