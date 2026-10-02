# feedback-masked-faults

Detectability and calibrated sequential detection of feedback-masked faults in
air-handling units.

Integral action holds the controlled variable at its setpoint under sensor bias
and capacity degradation, so these faults are invisible in the controlled
variable. Their signature moves into the control effort, relative to the effort
the current operating context requires. This repository contains:

- **C1**: the masking mechanism and the detection-delay theory for faults that
  are observable only through the control effort;
- **C2**: a sequential detector with a guaranteed false-alarm rate (conformal
  p-values, DKW correction, e-detector), evaluated on public AHU datasets.

## Setup

Requires Python >= 3.11.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Layout

| Path | Content |
|---|---|
| `src/fmf/sim/` | synthetic closed loop (PI + first-order plant + faults) |
| `src/fmf/theory/` | quasi-static masking relations, delay predictions, long-run variance |
| `src/fmf/detectors/` | CUSUM, conformal p-values with DKW correction, e-detectors |
| `src/fmf/metrics/` | run lengths, detection delay, ARL0 |
| `src/fmf/datasets/` | dataset loaders |
| `src/fmf/baselines/` | context models for the expected control effort |
| `src/fmf/rules/` | rule-based baselines (ASHRAE Guideline 36 AFDD, APAR) |
| `configs/` | one YAML file per experiment |
| `experiments/` | reproducible runs writing to `results/` and `paper/figs/` |
| `scripts/download/` | one download script per dataset, with SHA-256 checksums |
| `docs/decisions/` | architecture decision records |
| `paper/` | IFAC manuscript |

## Data

No data is committed. Each dataset is fetched into `data/raw/` by its script in
`scripts/download/`, which verifies SHA-256 checksums.

## Reproducing the paper

```bash
make figures   # experiments/fig_*.py -> paper/figs/
make tables    # experiments/tab_*.py -> results/tables/
make paper     # latexmk -pdf
```
