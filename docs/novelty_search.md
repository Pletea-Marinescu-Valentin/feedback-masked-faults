# Novelty search

Date: 2026-10-04. Scope: the two SAFEPROCESS contributions (C1 detectability
and delay of feedback-masked faults; C2 calibrated sequential detection).

## Protocol

- Databases: OpenAlex (primary; titles, abstracts and available full texts)
  and the arXiv API. Semantic Scholar was queried but returned HTTP 429 on
  every request without an API key. Scopus and Web of Science were not
  accessible from this environment; the authors should repeat the key queries
  there with institutional access before submission.
- Queries: 13 OpenAlex queries (`scripts/literature/search.py`, raw results in
  `results/literature/`, not committed) and 4 arXiv queries:
  - C1: fault masking + control signal; closed-loop + integral action + sensor
    bias; control-effort residual; valve-position residual in AHUs;
    closed-loop detectability + Kullback-Leibler; detection delay + CUSUM +
    long-run variance; CUSUM + AHU; seasonal detectability in HVAC.
  - C2: conformal prediction + fault detection + false alarm; e-values /
    e-detectors + fault detection; conformal change detection + process
    monitoring; conformal anomaly detection + HVAC; distribution-free
    false-alarm guarantee + fault detection; arXiv: conformal + change
    detection, "e-detector", conformal + fault detection, masked + feedback +
    fault detection.
- Screening: the 25 most relevant OpenAlex hits per query by title; abstracts
  read (`scripts/literature/abstracts.py`) for 15 candidates.

## Closest prior work

| Work | Relation | Difference from this paper |
|---|---|---|
| Salsbury and Diamond (2001), E&B 33(4) | monitors the PI compensation against a feedforward model | no information-rate or delay analysis, no calibrated false-alarm rate |
| Seem and House (2006, IFAC; 2009, HVAC&R Res.) | integrated control and fault detection for AHUs | idem |
| Sconyers et al. (2013), IJPHM 4(1) | detectors for fault modes masked by control loops, with prescribed confidence and false-alarm rate (hovercraft, simulation) | not HVAC; no tuning-invariant information rate, delay law or distribution-free sequential guarantee |
| Baldo et al. (2026), PHME | control energy vs. tracking error to reveal drifts hidden by feedback (simulation) | no detectability theory, no calibration guarantee, no real data |
| Bartyś (2021), ACTA IMEKO; Niemann and Stoustrup (1997), CDC; Wang, Chen and Song (2017), JPC; Zhang et al. (2026), LNNS | closed-loop effects on fault detection and isolation | general FDI, no masked-fault information rate or HVAC data |
| Chahine and Noura (2026), Sensors | CV-based virtual sensors fail on biased sensors pulled back by the loop | empirical, no effort-based detector |
| Kundacina et al. (2025), IEEE Access; Bates et al. (2023), Ann. Stat. | calibration-conditional conformal p-values (DKW, Simes) | per-sample detection, not sequential |
| Diallo, Homri and Dantan (2025), JPC; Mei et al. (2025, arXiv); Burger (2025, arXiv) | conformal thresholds, prediction sets or charts for fault detection / SPC | not sequential with an ARL guarantee |
| Vovk, Nouretdinov and Gammerman (2024, arXiv); Shin, Ramdas and Rinaldo (2023), NEJSDS; Ramdas (2026, arXiv); Bhattacharyya and Ramdas (2026, arXiv) | conformal CUSUM, e-detectors and their theory | general theory; no fixed calibration set with DKW correction, no minimum-delay design rule, no HVAC data |
| Han and Qu (2026, arXiv) | conformal-martingale monitors fire on 135 of 135 clean real forecast streams | same failure mode as our minute-level results, in ML forecasting; no block-scale remedy or delay trade-off |
| Shang et al. (2021, arXiv); Feng et al. (2024, arXiv) | distributionally robust fault detection with chance constraints on false alarms | per-sample design under moment ambiguity, not sequential |
| Dandapanthula and Ramdas (2025, arXiv) | error control across many monitored streams with e-detectors | relevant to the fleet discussion |

Not screened (abstract unavailable to the open tools): Guc (2026), "Conformal
Koopman residual monitoring: distribution-free, drift-resilient fault
detection for multivariate chemical processes", SSRN
doi:10.2139/ssrn.6989092. Check with institutional access.

## Conclusions for the manuscript

- Not claimable: that monitoring the control signal for masked faults is new;
  that conformal or e-value sequential detection is new; that such monitors
  fail on dependent real data (Han and Qu 2026 report it).
- Not found in the searched literature: the information rate
  m_eq^2 / (2 S) of the control-effort residual and its invariance to PI tuning
  and valve characteristic; the delay law (ln(I ARL0) - 1) / I with seasonal
  zero-information windows for masked faults; the minimum-delay bound of the
  calibration-conditional kappa-mixture e-detector; the block-scale study on
  real AHU data. The manuscript states these as contributions without
  priority claims.

## Prior results on the same datasets (2026-10-09)

Works citing Ghalamsiah et al. (2026) or using the real RP-1312 days, found
with OpenAlex (citations of doi:10.1038/s41597-025-06179-y; queries "ASHRAE
1312 air handling unit fault detection" and variants, 2022 onwards). Elsevier
abstracts were not available to the open tools.

| Work | Data | Reported result | Relation to this paper |
|---|---|---|---|
| Tra, Amayri and Bouguila (2024), IEEE T-ASE 21(4) | real RP-1312 days | per-sample F1 98.10 / 93.50 / 81.57 % (summer / winter / spring, 10 % missing data), unsupervised | public code (github.com/viettra-xai/DV-MPPCA): threshold = median of the test scores, random sample-level train/test split; no false-alarm rate on healthy units |
| Jiang (2026), Energy and Buildings 372 | six AHU domains of Ghalamsiah et al., incl. G36-Cyber and G36-HIL | macro-F1 0.790-0.811 within, at most 0.436 / 0.498 on Cyber / HIL | supervised cross-domain classification |
| Huang et al. (2026, SSRN); Rajabi and McArthur (2026, SSRN); Bi et al. (2026), Applied Energy 426 | simulation-to-real or augmentation on HVAC datasets | no comparable numbers in the abstracts | not cited |

None reports alarm rates on fault-free days or detection delays, so the
comparison in the paper is on the protocol (thresholds fixed in advance,
fault-free twin unit as ground truth) and on our own baselines run on the
same days, not on F1.

### Full texts read (2026-10-09)

The authors supplied the full texts; findings that change or refine the table above:

- Tra et al. (2024): the paper itself (Sec. IV-B) states the random 50/50
  split and a threshold set so that about half of the test instances are
  predicted faulty; it also excludes the control signals and uses only the
  122 sensor values.
- Jiang (2026): internal domains RBC-5wk, G36-1wk, G36-5wk and G36-Degrad;
  external G36-Cyber and G36-HIL (no RP-1312 real data). Reports 135-865
  false alarms per 1000 normal windows on Cyber and 0-324 on HIL, traded
  against 54-925 missed faults per 1000 fault windows.
- Yang et al. (2022), SETA 52: moving-window PCA with 95 % SPE limits on 52
  real fault-free RP-1312 days; daily share of samples with a fault symptom
  below 10 % on 31 days, 10-47 % on the other 21. Closest false-alarm
  comparison on the same real days. Cited.
- Gao, Li and Liang (2025), AEI 65: per-sample FAR 0.59-1.55 % (T2/SPE,
  GLR, ICA, CCA variants) on sensor faults of 10-15 % of the mean injected
  offline into AHU-B data, so not masked by the loop. Cited.
- Zhang et al. (2022), E&B 269, and Zhang et al. (2024), JOBE 96: report
  detection rates and times only (test sets of 120 normal samples followed by
  fault samples); no false-alarm rate. Not cited (space).
- Diallo et al. (2025): Tennessee Eastman; classical T2 (F distribution) and
  SPE limits, and per-sample conformal thresholds with the DKW-Massart
  correction following Bates et al. (2023). Confirms both uses in the paper.
- Prabhu Bam et al. (2024), E&B 313: supervised Neyman-Pearson classifiers on
  RP-1043 and RP-1312 that bound a per-sample error with high probability;
  needs labeled faults. Not cited (space).
- Bi et al. (2026), Applied Energy 426: supervised augmentation, RP-1043 and
  RP-1312, accuracy and F1. Wan et al. (2026), JOBE 128: LBNL SDAHU data.
  Not cited.

No paper changes the method: none reports alarm rates over time, detection
delay against a guaranteed ARL0, or residuals of the control effort for
masked faults.
