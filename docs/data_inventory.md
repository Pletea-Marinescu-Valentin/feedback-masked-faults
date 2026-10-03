# Data inventory

Checked on 2026-10-03 against the downloaded files (`make data`) and each
dataset's own documentation. Column names are those of the CSV headers.
"u" lists actuator commands, "z" context points, "CV" controlled variables.

## Ghalamsiah et al. (2026) — `scripts/download/ghalamsiah2026.py`

figshare doi:10.6084/m9.figshare.29297999.v3, one zip (928.7 MB, MD5 published).
Eight AHU datasets; documentation in each folder's `00_explanations.pdf`,
`conventions.pdf`, `DomainPairs.pdf`.

| Set | Type | Resolution | Duration | Fault-free data |
|---|---|---|---|---|
| 01 RBC-ASHRAE1312, Real | real (ERS, Iowa), RBC | 1 min | 49 test days: 18 summer 2007, 12 winter 2008, 19 spring 2008 | `baseline.csv` of every test day: the fault-free twin AHU on the same day (49 days) |
| 01 RBC-ASHRAE1312, Simulation | simulated | 1 min | same days | `baseline.csv` per day |
| 02 RBC-Nesbitt | real (Philadelphia), RBC | 5 min | 20 fault days, 2016-2018 | 22 baseline days (`Baselines/`), different dates from the fault days |
| 03 RBC-5wk | simulated, RBC, Tuscaloosa AL (21 faults) | 5 min | 5 weeks | `BaselineSystem.csv` |
| 04 G36-1wk | simulated, G36, Chicago (359 faults) | 1 min | one week per season | per season |
| 05 G36-5wk | simulated, G36, Tuscaloosa AL (3 faults) | 5 min | 5 weeks | yes |
| 06 G36-Degrad | simulated, G36, Tuscaloosa AL (12 faults) | 5 min | 1 year | `BaselineSystem.csv`, same weather and schedules as the fault runs |
| 07 G36-Cyber, 08 G36-HIL | simulated (Cyber) / hardware in the loop (HIL), Chicago | 5 min | 1 day | yes |

ERS (RBC-ASHRAE1312 Real), 162 columns, temperatures in °F:
- u: `CHWC-VLV` (% open), `HWC-VLV` (% **closed**), `SF-SPD` (%), `OA-DMPR`, `RA-DMPR`, `EA-DMPR`.
- CV: `SA-TEMP` / `SAT_SPT`, `SA-SP` / `SA_SPSPT`.
- z: `MA-TEMP`, `RA-TEMP`, `OA-TEMP`, `SA-CFM`, `CHWC-EWT`, `HWC-EWT`, `RA-HUMD`, `SYS-CTL` (occupancy).
- Fault files carry `OA-TEMP = 0`; the outdoor temperature comes from `weather.csv`.
- The faulted unit has no fault-free day in this release, so unit differences cannot be separated from fault effects on faulted days.
- Masked faults: `HeaCoiValLea*` (summer: the cooling valve compensates), `HeaCoiFou_*`, `HeatCoiReduCapa_*` (winter; stages 2-3 saturate and show in the CV), `AirFilBlock_*`, `SupDucLea_*`.

Nesbitt (RBC-Nesbitt), 540 columns, 5 min, three AHU column blocks with
repeated names (pandas suffixes `.1`, `.2`); the block-to-AHU mapping is not
stated in the headers.
- u: `Feedback: Cooling Coil Valve Position` (feedback, no command logged),
  `Control: Supply Fan Speed`, `Control: Steam Valve 1/3`, `2/3`, economizer position.
- Baseline days: cooling active on 12 (2016-06 to 2016-10, 2017-06/07), steam
  heating on most winter days; the measured supply-air temperature sits up to
  3.4 degF above its setpoint on several winter days.
- The supply-air "sensor bias" faults (`TSup_m4`, `TSup_p4`, `TSup_m3.5`) were
  emulated by overriding the setpoint demand adjust. On 2016-09-07 the measured
  supply-air temperature rises 3.4 degF above its logged setpoint while the
  valve barely moves, i.e. the fault shows in the controlled variable; on
  2016-08-08 the logged setpoint itself moves with the reset. Not usable as
  masked-sensor-bias cases without the AHU mapping and the emulation details.

G36-Degrad, 114 columns (151 header fields including zones), temperatures in K:
- u: cooling valve, heating valve, supply-fan speed signal, outdoor/return/exhaust damper signals.
- CV: supply-air temperature and setpoint, duct static pressure and setpoint.
- z: outdoor dry/wet bulb, mixed and return air temperature, supply and outdoor air mass flow, occupancy, chilled-water temperatures.
- Faults injected in occupied hours (weekdays 7-19): `CooCoiAirFou_hea7_pre30`, `_hea14_pre200` (air-side coil fouling, linear in time from 1 January), `TSup_p1`, `TSup_m1` (supply-air sensor bias growing to ±1 °C over the year), chiller condenser fouling (from day 170), duct leakage.
- Sensors are noise-free; paired differences fault - baseline show isolated hours where a tiny perturbation shifts a mode switch.

## LBNL 2019 (Granderson and Lin) — `scripts/download/lbnl2019.py`

OEDI submission 910, doi:10.25984/1824861; zip 6.7 MB plus `lbnl2019_inventory.pdf`
(owner-password protected; opens with an empty password). All sets at 1 min.

| File | Type | Fault-free days |
|---|---|---|
| `MZVAV-1.csv` | simulated (PNNL large office, Chicago) | unfaulted weeks in each season |
| `MZVAV-2-1.csv`, `MZVAV-2-2.csv` | ERS RP-1312, experimental and simulated; 19 normalized columns | 12 unfaulted days in 2008-2009 (`Fault Detection Ground Truth` = 0) |
| `SZCAV.csv`, `SZVAV.csv` | experimental, FLEXLAB cell X3A, same AHU in winter (CAV) and summer (VAV) 2017 | SZCAV 1 day (2017-04-01), SZVAV 4 days (2017-09-20, 21, 23, 24) |
| `RTU.csv` | experimental (ORNL FRP) | see inventory PDF |

The OEDI landing page describes the set as simulated; the inventory PDF states
that the FLEXLAB and ERS parts are experimental. In `MZVAV-2-1.csv` the day
2007-08-30 (heating-valve leak stage 3 in the inventory) is labeled 0.

## LBNL 2022 SDAHU (Granderson et al. 2023) — `scripts/download/lbnl2022.py`

figshare article 22338283 (collection 6486349), mirror of OEDI 5763; zip 608 MB
with `LBNL_FDD_Data_Sets_SDAHU.pdf` and a Brick model (`.ttl`, Brick 1.2).
Simulated single-duct VAV AHU, TMY Chicago, 1 min, 2018-01-01 01:00 to
2018-12-31 23:59 (525 540 rows), 31 columns, temperatures in °F.
- u: `CHWC_VLV` (0-1), `SF_SPD`, `RF_SPD`, `OA_DMPR`, `RA_DMPR` (and `_DM` demands).
- CV: `SA_TEMP` / `SA_TEMPSPT`, `SA_SP` / `SA_SPSPT`.
- z: `OA_TEMP`, `MA_TEMP`, `RA_TEMP`, `SA_CFM`, `OA_CFM`, `RA_CFM`, zone temperatures, `SYS_CTL`.
- Fault-free year `AHU_annual.csv`; 20 faults, each present for the whole year.
- The PDF lists supply-air sensor bias files as `sa_bias_±2/±4_annual.csv`; the
  archive contains `coi_bias_±2/±4_annual.csv` and no `sa_bias_*` file. TODO verify
  that these are the same scenarios.

## Wang (2025) — `scripts/download/wang2025.py`

figshare doi:10.6084/m9.figshare.27147678.v3, three CSVs (30 MB). Real operation,
41 CAV AHUs (office, hospital, auditorium), hourly aggregates of 1-min data,
labels from rule-based annotation (normal + six fault classes).
- u: `Valve position` (one valve per AHU), `Supply fan`.
- z: `Set point temperature`, `Return temperature`, plant supply temperatures and pump statuses.
- No outdoor-air, mixed-air temperature or airflow: a context model for the valve
  command has little to work with. Not used for C1/C2; candidate for a
  fleet-level false-alarm study only if a usable context can be built.

## Use in the SAFEPROCESS paper

| Purpose | Data |
|---|---|
| Masking and delay theory | synthetic loop (`fmf.sim`) |
| Seasonal information windows | G36-Degrad signatures + ERS residual noise |
| False alarms on real fault-free data | ERS twin-AHU baseline days |
| Delay on real noise | ERS test days with injected steps |
