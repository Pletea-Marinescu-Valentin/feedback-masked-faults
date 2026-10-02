"""Granderson et al. (2023), Sci. Data 10:342: LBNL FDD data sets, SDAHU part.

Source: figshare collection doi:10.6084/m9.figshare.c.6486349.v1, article
22338283 (mirror of OEDI submission 5763, doi:10.25984/1881324).
"""

from _common import Remote, fetch

DATASET = "lbnl2022"
FILES = [
    Remote("https://ndownloader.figshare.com/files/39742909", "LBNL_FDD_Dataset_SDAHU_all_3.zip",
           608146162, "f53a622b595c578a846a0d1988f8dc07"),
]

if __name__ == "__main__":
    for remote in FILES:
        fetch(DATASET, remote, extract=True)
