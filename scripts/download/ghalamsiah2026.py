"""Ghalamsiah et al. (2026), Sci. Data 13:15: eight labeled AHU datasets.

Source: figshare, doi:10.6084/m9.figshare.29297999.v3 (one archive with
RBC-ASHRAE1312, RBC-Nesbitt, RBC-5wk, G36-1wk, G36-5wk, G36-Degrad, G36-Cyber,
G36-HIL).
"""

from _common import Remote, fetch

DATASET = "ghalamsiah2026"
FILES = [
    Remote("https://ndownloader.figshare.com/files/57268553",
           "Public_ScientificData_AHUFaults.zip", 928675992,
           "5649a042ba05462ff01b1c32f5b7788b"),
]

if __name__ == "__main__":
    for remote in FILES:
        fetch(DATASET, remote, extract=True)
