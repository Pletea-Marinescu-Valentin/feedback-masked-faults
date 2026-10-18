# Data used in the paper

All data come from one public archive: Ghalamsiah et al. (2026), *Labeled
datasets for air handling units operating in faulted and fault-free states*,
Scientific Data 13:15, doi:10.1038/s41597-025-06179-y. The archive is on
figshare (doi:10.6084/m9.figshare.29297999.v3): one zip of 928.7 MB with a
published MD5, about 6.7 GB after extraction. `scripts/download/ghalamsiah2026.py`
fetches and verifies it.

The archive holds eight AHU datasets; the paper uses two of them. Documentation
is in each folder's `00_explanations.pdf`, `conventions.pdf` and `DomainPairs.pdf`.
Column names below are those of the CSV headers.

## RBC-ASHRAE1312, Real (ERS)

Real data from two identical AHUs at the Iowa Energy Resource Station
(ASHRAE RP-1312), rule-based control, 1-min resolution, 162 columns,
temperatures in °F.

- Test days: 49, of which 18 are in summer 2007, 12 in winter 2008 and 19 in spring 2008.
  On every day one unit carries a fault and the twin unit runs fault-free
  (`baseline.csv`), so the twin unit gives 49 real fault-free days.
- Control effort u: `CHWC-VLV` (cooling valve, % open), `HWC-VLV` (heating valve,
  % **closed**), `SF-SPD` (supply fan, %), `OA-DMPR`, `RA-DMPR`, `EA-DMPR`.
- Controlled variables (CV): `SA-TEMP` / `SAT_SPT`, `SA-SP` / `SA_SPSPT`.
- Context z: `MA-TEMP`, `RA-TEMP`, `OA-TEMP`, `SA-CFM`, `CHWC-EWT`, `HWC-EWT`,
  `RA-HUMD`, `SYS-CTL` (occupancy).
- Fault files carry `OA-TEMP = 0`; the outdoor temperature comes from `weather.csv`.
- The faulted unit has no fault-free day in this release, so differences between
  the units cannot be separated from fault effects on faulted days.
- Masked faults: `HeaCoiValLea*` (summer: the cooling valve compensates),
  `HeaCoiFou_*`, `HeatCoiReduCapa_*` (winter; stages 2-3 saturate and show in the
  CV), `AirFilBlock_*`, `SupDucLea_*`.

## G36-Degrad

Simulated year of a Guideline 36 AHU (Tuscaloosa, AL), 5-min resolution,
114 columns (151 header fields including zones), temperatures in K.

- Control effort u: cooling valve, heating valve, supply-fan speed signal,
  outdoor/return/exhaust damper signals.
- CV: supply-air temperature and setpoint, duct static pressure and setpoint.
- Context z: outdoor dry/wet bulb, mixed and return air temperature, supply and
  outdoor air mass flow, occupancy, chilled-water temperatures.
- Fault-free reference: `BaselineSystem.csv`, with the same weather and schedules
  as the fault runs.
- Faults are injected in occupied hours (weekdays 7-19). The paper uses
  `CooCoiAirFou_hea14_pre200.csv` (air-side coil fouling, linear in time from
  1 January) and `TSup_p1.csv` (supply-air sensor bias growing to +1 °C over the
  year); the loader also maps `CooCoiAirFou_hea7_pre30.csv` and `TSup_m1.csv`.
- Sensors are noise-free. Paired differences fault − baseline show isolated hours
  where a tiny perturbation shifts a mode switch.

## Use in the paper

| Purpose | Data |
|---|---|
| Masking and delay theory | synthetic loop (`fmf.sim`), no download needed |
| Seasonal information windows | G36-Degrad signatures + ERS residual noise |
| False alarms on real fault-free data | ERS twin-AHU baseline days |
| Delay on real noise | ERS test days with injected steps |
