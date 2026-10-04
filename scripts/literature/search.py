"""Literature search for the novelty check (OpenAlex and Semantic Scholar).

Runs a fixed list of queries, stores the raw hits in results/literature/ and
prints title, year, venue and DOI for screening. OpenAlex searches titles,
abstracts and available full texts; Semantic Scholar searches titles and
abstracts. Neither replaces Scopus or Web of Science; both are open indexes
with broad overlap.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "literature"
HEADERS = {"User-Agent": "fmf-literature-search"}

QUERIES = {
    # C1: information left by feedback-masked faults in the control effort
    "c1_masking_control_signal": '"fault masking" feedback control signal detection',
    "c1_closed_loop_integral": 'closed-loop fault detection integral action sensor bias control signal',
    "c1_control_effort_residual": '"control effort" residual fault detection',
    "c1_valve_residual_ahu": 'valve position residual fault detection air handling unit',
    "c1_detectability_closed_loop": 'fault detectability closed-loop feedback information Kullback-Leibler',
    "c1_detection_delay_spectral": 'detection delay CUSUM long-run variance autocorrelated residuals',
    "c1_cusum_ahu": 'CUSUM air handling unit fault detection',
    "c1_seasonal_detectability": 'seasonal fault detectability HVAC operating conditions',
    # C2: calibrated sequential detection
    "c2_conformal_fault_detection": 'conformal prediction fault detection false alarm',
    "c2_e_values_fault": 'e-values e-detector fault detection monitoring',
    "c2_conformal_change_detection": 'conformal change detection process monitoring',
    "c2_conformal_hvac": 'conformal anomaly detection building HVAC',
    "c2_false_alarm_guarantee": 'distribution-free false alarm guarantee fault detection',
}


def get(url: str) -> dict:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def openalex(query: str, n: int = 25) -> dict:
    url = ("https://api.openalex.org/works?per-page=%d&search=%s"
           "&select=id,doi,title,publication_year,primary_location,cited_by_count"
           % (n, urllib.parse.quote(query)))
    return get(url)


def semantic_scholar(query: str, n: int = 25) -> dict:
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?limit=%d&query=%s"
           "&fields=title,year,venue,externalIds,citationCount" % (n, urllib.parse.quote(query)))
    return get(url)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = [a for a in sys.argv[1:] if not a.startswith("--")] or list(QUERIES)
    for key in keys:
        query = QUERIES[key]
        oa = openalex(query)
        time.sleep(1.0)
        ss = {"error": "skipped"}
        if "--s2" in sys.argv:
            try:
                ss = semantic_scholar(query)
            except Exception as exc:  # rate limits are frequent without an API key
                ss = {"error": str(exc)}
            time.sleep(12.0)
        (OUT / f"{key}.json").write_text(json.dumps({"query": query, "openalex": oa,
                                                     "semantic_scholar": ss}, indent=1))
        print(f"\n=== {key}: {query}")
        print(f"  OpenAlex hits: {oa['meta']['count']}")
        for w in oa["results"]:
            venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
            print(f"   [{w['publication_year']}] {w['title']} | {venue} | {w.get('doi')}")
        if "data" in ss:
            print(f"  Semantic Scholar hits: {ss.get('total')}")
            for p in ss["data"]:
                doi = (p.get("externalIds") or {}).get("DOI")
                print(f"   [{p.get('year')}] {p.get('title')} | {p.get('venue')} | {doi}")
        else:
            print(f"  Semantic Scholar: {ss.get('error')}")


if __name__ == "__main__":
    main()
