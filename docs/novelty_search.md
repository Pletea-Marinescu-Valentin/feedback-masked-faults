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
