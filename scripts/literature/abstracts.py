"""Print OpenAlex abstracts (reconstructed from the inverted index) for given DOIs."""

import json
import sys
import urllib.parse
import urllib.request

HEADERS = {"User-Agent": "fmf-literature-search"}


def abstract(inverted: dict | None) -> str:
    if not inverted:
        return "(no abstract in OpenAlex)"
    words = sorted((pos, word) for word, positions in inverted.items() for pos in positions)
    return " ".join(word for _, word in words)


for doi in sys.argv[1:]:
    url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi)
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        w = json.load(r)
    authors = ", ".join(a["author"]["display_name"] for a in w.get("authorships", [])[:6])
    venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
    print(f"=== {w['title']} ({w['publication_year']}) | {venue} | {authors} | cited {w.get('cited_by_count')}")
    print(abstract(w.get("abstract_inverted_index")))
    print()
