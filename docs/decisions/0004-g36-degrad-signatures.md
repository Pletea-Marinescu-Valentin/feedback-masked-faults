# 0004. Fault signatures from G36-Degrad

- Status: accepted
- Date: 2026-10-03

## Context

G36-Degrad simulates one year fault-free and once per degradation scenario
with identical weather and schedules, but with noise-free sensors. Paired
differences of the commands contain isolated hours in which a tiny
perturbation shifts a discrete mode switch.

## Decision

- Signature: hourly means of u_fault - u_baseline over occupied samples, then
  the median of the same hour of day within each calendar week.
- Noise: hourly spread of the out-of-fold residual of the same actuator on all
  real ERS fault-free days of the relevant seasons; hourly blocks treated as
  independent.
- Detectors: CUSUM matched to the signature, threshold ln(ARL0) with ARL0 one
  operating year (260 days x 12 h); 200 noise replicas.

## Consequences

The figure combines simulated signatures with real noise levels; it does not
use the simulator's own (unrealistically small) residual noise. Day-level
correlation of the real noise is not reproduced.
