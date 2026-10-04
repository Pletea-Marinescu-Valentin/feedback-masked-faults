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
| `src/fmf/rules/` | Guideline 36 AFDD fault conditions (Addendum p to G36-2021) |
| `configs/` | one YAML file per experiment |
| `experiments/` | reproducible runs writing to `results/` and `paper/figs/` |
| `scripts/download/` | one download script per dataset, with SHA-256 checksums |
| `docs/decisions/` | architecture decision records |
| `paper/` | IFAC manuscript |

## Data

No data is committed. `make data` (or the scripts in `scripts/download/`)
fetches the four public datasets into `data/raw/`, verifying sizes, provider
MD5s and the SHA-256 digests pinned in `scripts/download/SHA256SUMS`
(about 1.5 GB of archives, 10 GB on disk after extraction). `docs/data_inventory.md` lists
the points, resolutions and fault-free periods of each set.

## Experiments

| Script | Paper item | Data | Run time |
|---|---|---|---|
| `experiments/fig_masking.py` | Fig. 1 | synthetic loop | seconds |
| `experiments/fig_delay_theory.py` | Fig. 2 | synthetic loop, 40 conditions | about 2 min |
| `experiments/fig_seasonal.py` | Fig. 3 | G36-Degrad + ERS noise | about 30 s |
| `experiments/tab_far_ers.py` | Table 1 | ERS fault-free days | about 15 s |
| `experiments/fig_delay_arl.py` | Fig. 4 | ERS residual noise | seconds |
| `experiments/tab_detection_ers.py` | text (real faults) | ERS faulted days | about 1 min |
| `experiments/tab_design.py` | text (fleet budget, minimum delay) | formulas only | instant |
| `experiments/tab_guideline_ers.py` | text (Guideline 36 AFDD baseline) | ERS days | seconds |

Each script reads `configs/<name>.yaml` (fixed seeds), writes intermediate
results to `results/`, figures to `paper/figs/`, and every number quoted in the
paper to `paper/generated/<name>.tex`. Scripts with a long simulation accept
`--plot-only` (or `--tables-only`) to redraw from `results/`.
Design decisions that change results are recorded in `docs/decisions/`.

## Reproducing the paper

```bash
make data      # download and verify the datasets
make test      # unit tests
make figures   # experiments/fig_*.py -> paper/figs/, paper/generated/
make tables    # experiments/tab_*.py -> results/tables/, paper/generated/
make paper     # latexmk -pdf (needs the IFAC class, see paper/README.md)
```
