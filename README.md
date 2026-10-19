# feedback-masked-faults

Code and reproduction pipeline for the paper *Detectability and Calibrated
Sequential Detection of Feedback-Masked Faults in Air-Handling Units*
(submitted to IFAC SAFEPROCESS 2027).

Integral action holds the controlled variable of a control loop at its setpoint
when a sensor drifts or a component loses capacity, so the fault signature
moves into the control effort. The paper (i) quantifies how fast such faults
can be detected from the control-effort residual and (ii) monitors that
residual with a sequential detector whose false-alarm rate is guaranteed
(conformal p-values, a Dvoretzky–Kiefer–Wolfowitz correction and e-detectors).

**Every figure and every number in the paper is produced by a script in this
repository.** This README walks you through running the whole pipeline and
checking that you obtain exactly the same results.

---

## What you need

| | |
|---|---|
| Python | 3.11 or newer. The paper was produced with **Python 3.13.12** on Windows 11 (x86-64). |
| Disk | about 8 GB free (a 929 MB archive, 6.7 GB after extraction) |
| Network | to download the public dataset once |
| Time | about 15 minutes in total: download, ~10 s of tests, ~6 min of experiments |
| `make` | optional; every step below also has a plain command |

No GPU, no commercial software and no account are needed.

---

## Step 1: get the code

```bash
git clone https://github.com/Pletea-Marinescu-Valentin/feedback-masked-faults.git
cd feedback-masked-faults
```

## Step 2: create the Python environment

Install the exact package versions used for the paper from
`requirements-lock.txt`, then the project itself.

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

All later commands assume this environment is active.

## Step 3: run the unit tests

```bash
pytest            # or: make test
```

Expected: `97 passed` in about 10 seconds. The tests run on synthetic data and
check the building blocks against theory, among others:

- the e-detector's empirical average run length to false alarm is at least 1/α
  on i.i.d. data;
- the CUSUM delay matches ln(ARL₀)/I for a known Gaussian shift;
- the AR(1) variance-inflation factor (1+ρ)/(1−ρ);
- the masking relations of the synthetic closed loop;
- the Guideline 36 fault conditions and the PCA monitor.

## Step 4: download the data

```bash
python scripts/download/ghalamsiah2026.py     # or: make data
```

This fetches one public archive, Ghalamsiah et al. (2026), Scientific Data
13:15, from figshare (doi:10.6084/m9.figshare.29297999.v3). The script:

1. downloads `Public_ScientificData_AHUFaults.zip` (928.7 MB, resumable) into `data/raw/ghalamsiah2026/`;
2. checks its size, the MD5 published by figshare and the SHA-256 pinned in
   `scripts/download/SHA256SUMS`, and stops if any of them differs;
3. extracts it next to the archive.

The paper uses two of the eight datasets in the archive: the real ASHRAE
RP-1312 data of the Iowa Energy Resource Station (ERS) and the simulated
G36-Degrad year. `docs/data_inventory.md` lists the columns and periods used.
Nothing in `data/` is committed to this repository.

## Step 5: run the experiments

With `make`:

```bash
make figures tables
```

Without `make`, Linux / macOS:

```bash
for s in experiments/fig_*.py experiments/tab_*.py; do echo "$s"; python "$s" || break; done
```

Without `make`, Windows (PowerShell):

```powershell
Get-ChildItem experiments\fig_*.py, experiments\tab_*.py | ForEach-Object { $_.Name; python $_.FullName }
```

Each script reads its settings and fixed random seed from `configs/<name>.yaml`
and writes:

- figures to `paper/figs/<name>.pdf`;
- result tables to `results/tables/<name>.csv`;
- intermediate results to `results/<name>.json`;
- every number quoted in the paper to `paper/generated/<name>.tex`, as LaTeX macros.

| Script | Paper item | Data | Run time* |
|---|---|---|---|
| `experiments/fig_masking.py` | Fig. 1, masking on a synthetic loop | synthetic | 2 s |
| `experiments/fig_delay_theory.py` | Fig. 2, predicted vs. measured information and delay | synthetic, 40 conditions | 2 min 45 s |
| `experiments/fig_seasonal.py` | Fig. 3, seasonal information windows | G36-Degrad + ERS noise | 35 s |
| `experiments/fig_delay_arl.py` | Fig. 4, mean delay vs. nominal ARL₀ | ERS residual noise | 10 s |
| `experiments/tab_far_ers.py` | Table 1, false alarms on real fault-free days | ERS | 12 s |
| `experiments/tab_detection_ers.py` | Sec. 5, real faulted and fault-free units | ERS | 1 min 40 s |
| `experiments/tab_guideline_ers.py` | Sec. 5, Guideline 36 AFDD baseline | ERS | 13 s |
| `experiments/tab_pca_ers.py` | Sec. 5, PCA T²/SPE baseline | ERS | 4 s |
| `experiments/tab_design.py` | Sec. 6, fleet alarm budget and minimum delay | formulas only | 1 s |

\* Measured on a 16-thread x86-64 laptop; about 6 minutes in total.

The two longest scripts accept `--plot-only` to redraw a figure from
`results/<name>.json` without recomputing it.

## Step 6: check that you obtained the same results

```bash
python scripts/verify_numbers.py     # or: make verify
```

The script compares every number your run wrote to `paper/generated/` with the
values printed in the submitted paper, stored in
`results/tables/paper_numbers.csv`. The expected output is:

```
188 of 188 numbers match the paper.
```

Any difference is listed with the macro name, the value in the paper and your
value, and the script exits with status 1.

The result tables and figures are also tracked in git, so

```bash
git status --short results/tables paper/figs
```

should print nothing: your run rewrote the files with identical content. The
figure PDFs can differ byte-wise if your system lacks the Times New Roman font,
because matplotlib then falls back to another serif font. The numbers are not
affected.

`make reproduce` runs Steps 3, 5 and 6 in one go.

---

## Determinism

- **Seeds.** All randomness is seeded from the YAML files in `configs/`.
- **Thread count.** Results do not depend on the number of threads. We
  obtained identical numbers with 1, 3 and 16 threads, for example with
  `OMP_NUM_THREADS=1`.
- **Package versions.** Gradient-boosted trees (LightGBM) and numerical libraries can
  change results slightly between versions. Use `requirements-lock.txt` for an
  exact match.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'fmf'` | The project is not installed in the active environment. Activate `.venv` and run `pip install -e . --no-deps`. |
| `FileNotFoundError` under `data/raw/` | Step 4 has not finished. Re-run the download script; it resumes a partial download. |
| The download stops with a checksum error | The archive is corrupt or has changed on figshare. Delete `data/raw/ghalamsiah2026/` and download again. |
| `MISSING <experiment>` from `verify_numbers.py` | That experiment has not run yet; run `experiments/<experiment>.py`. |
| A few numbers differ in the last digit | The package versions differ from `requirements-lock.txt`. |

---

## How the code is organized

| Path | Content |
|---|---|
| `src/fmf/sim/` | synthetic closed loop: PI controller, first-order plant, valve, faults, noise |
| `src/fmf/theory/` | masking relations, information rate and delay law, long-run variance |
| `src/fmf/detectors/` | Gaussian CUSUM, conformal p-values with DKW correction, e-detectors, prewhitening, block batching |
| `src/fmf/baselines/` | context model of the expected control effort (LightGBM, out-of-fold by day), PCA monitor |
| `src/fmf/rules/` | Guideline 36 AFDD fault conditions (Addendum p to G36-2021) |
| `src/fmf/metrics/` | run lengths, detection delay, ARL₀ |
| `src/fmf/datasets/` | loader for the ERS and G36-Degrad data |
| `configs/` | one YAML file per experiment, with all parameters and seeds |
| `experiments/` | one script per figure or table of the paper |
| `scripts/download/` | dataset download with MD5 and SHA-256 verification |
| `scripts/verify_numbers.py` | comparison of your numbers with the paper |
| `results/tables/` | result tables and the reference numbers of the paper |
| `paper/figs/` | the figures of the paper |
| `docs/data_inventory.md` | columns, resolutions and periods of the data used |
| `docs/decisions/` | short records of every choice that affects results (data split, detector design, minimum delay, baselines) |
| `tests/` | unit tests on synthetic data |

The manuscript source is not part of this repository; `make paper` only works
with it.

## Data source

Ghalamsiah, N., Wen, J., Li, G., Chen, Y., Lu, X., and Fu, Y. (2026). Labeled
datasets for air handling units operating in faulted and fault-free states.
*Scientific Data*, 13, 15. doi:10.1038/s41597-025-06179-y. Data:
doi:10.6084/m9.figshare.29297999.v3. Please cite the dataset if you use it.
