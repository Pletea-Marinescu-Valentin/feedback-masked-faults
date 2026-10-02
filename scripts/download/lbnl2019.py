"""Granderson and Lin (2019): data sets for evaluation of building FDD algorithms.

Source: OEDI submission 910, doi:10.25984/1824861. Contains the simulated
MZVAV set (PNNL), ASHRAE RP-1312 (ERS, experimental and simulated), the
FLEXLAB SZCAV/SZVAV experiments and an RTU set, plus the inventory PDF.
OEDI publishes no checksums; SHA-256 values are pinned on first download.
"""

from _common import Remote, fetch

DATASET = "lbnl2019"
BASE = "https://data.openei.org/files/910/"
FILES = [
    Remote(BASE + "Data%20Sets%20for%20AFDD%20Evauluation%20of%20Building%20FDD%20Algorithms.zip",
           "lbnl2019_afdd_datasets.zip"),
    Remote(BASE + "lbnldatasynthesisinventory.pdf", "lbnl2019_inventory.pdf"),
]

if __name__ == "__main__":
    for remote in FILES:
        fetch(DATASET, remote, extract=True)
