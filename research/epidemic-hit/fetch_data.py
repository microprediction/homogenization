"""Download NYT US county COVID-19 cumulative cases and deaths for 2020-21 (github.com/nytimes/covid-19-data) and
Census 2019 county population estimates into data/."""
import pathlib, urllib.request
HERE = pathlib.Path(__file__).resolve().parent
URLS = {"us-counties-2020.csv": "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2020.csv",
        "us-counties-2021.csv": "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2021.csv",
        "co-est2019-alldata.csv": "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv"}
if __name__ == "__main__":
    (HERE / "data").mkdir(exist_ok=True)
    for name, url in URLS.items():
        if not (HERE / "data" / name).exists():
            urllib.request.urlretrieve(url, HERE / "data" / name)
        print(name, (HERE / "data" / name).stat().st_size)
