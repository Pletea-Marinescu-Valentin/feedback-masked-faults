# 0005. Guideline 36 AFDD baseline

- Status: accepted
- Date: 2026-10-04

## Context

CLAUDE.md lists the Guideline 36 AFDD rules as a baseline. The guideline is
not freely available; Addendum p to Guideline 36-2021 (ASHRAE, 2024), a free
official addendum, restates the multiple-zone VAV AFDD section 5.16.14 with
the operating-state tables, the fault conditions FC#1-15 and the default
internal variables. The rules of APAR (Schein et al., 2006) are their
predecessor; the addendum states that the defaults derive from NISTIR 7365.

## Decision

- Implementation in `fmf.rules.g36_afdd` follows Addendum p: operating states
  from commanded positions (single common minimum-OA/economizer damper table),
  five-minute rolling averages at one-minute sampling, default thresholds,
  applicable states per condition, ModeDelay 30 min after the start of
  occupancy, AlarmDelay 30 min.
- Equalities in the state table use a tolerance of 0.01 on positions.
- MinOA-P per season is the most frequent outdoor-air damper position below
  99 % on the fault-free days (0.40 in summer, 0.50 in winter at the ERS).
- FC#6 is not evaluated: the active minimum outdoor-air setpoint is not logged.
- Duct-pressure threshold converted to the dataset unit (0.1 in. w.c.).
- Two sensor sets: the ERS dedicated coil-discharge sensors, and MAT/SAT only
  with the supply fan between them, as on most AHUs.
- APAR is not implemented separately.

## Consequences

The baseline uses default, untuned thresholds, as the guideline ships them;
the guideline itself recommends field tuning. Alarms on fault-free days may
include real capacity limits on hot days; the fault-free label of the twin
unit is the only ground truth available.
