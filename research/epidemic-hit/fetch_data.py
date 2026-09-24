"""Download NYT US county COVID-19 cumulative cases and deaths for 2020 (github.com/nytimes/covid-19-data) into data/."""
import pathlib, urllib.request
HERE = pathlib.Path(__file__).resolve().parent
URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2020.csv"
if __name__ == "__main__":
    (HERE / "data").mkdir(exist_ok=True)
    urllib.request.urlretrieve(URL, HERE / "data" / "us-counties-2020.csv")
    print((HERE / "data" / "us-counties-2020.csv").stat().st_size)
