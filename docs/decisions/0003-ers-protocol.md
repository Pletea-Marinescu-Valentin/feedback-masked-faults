# 0003. Protocol for the real ERS (RP-1312) data

- Status: accepted
- Date: 2026-10-03

## Context

Ghalamsiah et al. (2026) provide, for each of 49 ERS test days, the faulted
AHU and the fault-free twin AHU on the same day (1 min). The faulted unit has
no fault-free days in this release.

## Decision

- False-alarm study on the fault-free twin only. Within each season, days are
  split chronologically: the first half (rounded up) calibrates, the second
  half tests. No shuffling across time.
- Samples: occupied and supply fan on. Monitored commands: cooling valve
  (summer, spring), heating valve (winter), supply fan (all seasons).
- Context model: LightGBM (300 trees, learning rate 0.05, 15 leaves, 50 samples
  per leaf, subsampling 0.8) on mixed, outdoor, return air temperature, supply
  setpoint, supply airflow, coil entering water temperatures, return humidity
  and hour of day. Calibration residuals are out of fold by day; test residuals
  come from the model fitted on all calibration days.
- Outdoor temperature from `weather.csv` (fault files carry zeros).
- Faulted days are evaluated with detectors calibrated on all twin days and
  reported only as an illustration, since unit differences are confounded
  with fault effects.

## Consequences

Test sets are small (6-24 days): false-alarm counts are low and rates at
ARL0 = 30 days rest on few events. Seasonal shift between calibration and test
is not covered by this split.
