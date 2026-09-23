"""Download the Keith & Nitu (2023) spin-down time series (Zenodo 10.5281/zenodo.7664166, CC-BY 4.0) into data/."""
import io, pathlib, urllib.request, zipfile

HERE = pathlib.Path(__file__).resolve().parent
URL = "https://zenodo.org/api/records/7664166/files/nudot_timeseries.zip/content"

if __name__ == "__main__":
    raw = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "homogenization-research"}), timeout=120).read()
    zipfile.ZipFile(io.BytesIO(raw)).extractall(HERE / "data")
    for p in sorted((HERE / "data").rglob("*")):
        if p.is_file():
            print(p.relative_to(HERE), p.stat().st_size)
