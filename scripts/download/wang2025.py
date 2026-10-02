"""Wang (2025), Sci. Data 12:1481: real operational labeled AHU data (hourly).

Source: figshare, doi:10.6084/m9.figshare.27147678.v3.
"""

from _common import Remote, fetch

DATASET = "wang2025"
FILES = [
    Remote("https://ndownloader.figshare.com/files/53483432", "office_scientific_data.csv",
           15414979, "646799c043c8d0455d507cea223d370f"),
    Remote("https://ndownloader.figshare.com/files/53483435", "hosptial_scientific_data.csv",
           5621617, "3be98fc565727053878c205de03ffb4f"),
    Remote("https://ndownloader.figshare.com/files/53483438", "auditorium_scientific_data.csv",
           9147313, "044c9faf5d99d7621edd7fb05850dfd0"),
]

if __name__ == "__main__":
    for remote in FILES:
        fetch(DATASET, remote)
