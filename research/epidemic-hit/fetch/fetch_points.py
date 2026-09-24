"""Download the primary-source files behind the three point checks (Geneva, England regions, Manaus) and the
OWID national subset into raw/. Idempotent: files already present are kept. Run: python fetch_points.py [part ...]
with parts from: swiss, uk, papers, brazil, sim, owid (default: all).

One file cannot be fetched by script: the Science supplementary PDF for Buss et al. (2021) sits behind bot
protection. It was saved through a browser session to raw/buss_science/abe9728_buss_sm.pdf (see SOURCES.md)."""
import csv, gzip, io, pathlib, sys, urllib.parse, urllib.request, zipfile

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
UA = {"User-Agent": "Mozilla/5.0 (research data fetch; peter.cotton research project)"}


def get(url, dest, timeout=600):
    dest = RAW / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print("have", dest.relative_to(HERE), dest.stat().st_size)
        return dest
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        src = gzip.GzipFile(fileobj=r) if r.headers.get("Content-Encoding") == "gzip" else r
        while chunk := src.read(1 << 20):
            f.write(chunk)
    print("got ", dest.relative_to(HERE), dest.stat().st_size)
    return dest


# ---------------------------------------------------------------- Switzerland / Geneva
FOPH = "https://covid19.admin.ch/api/data/20231206-0sxi4s4a/sources/"  # final FOPH data release (Dec 2023)


def swiss():
    get("https://raw.githubusercontent.com/openZH/covid_19/master/COVID19_Fallzahlen_CH_total_v2.csv",
        "COVID19_Fallzahlen_CH_total_v2.csv")
    get(FOPH + "COVID19Death_geoRegion.csv", "foph_COVID19Death_geoRegion.csv")
    get(FOPH + "COVID19Hosp_geoRegion.csv", "foph_COVID19Hosp_geoRegion.csv")
    base = "https://raw.githubusercontent.com/HopkinsIDD/sarscov2-ifr-gva/master/"
    for f in ["data/stratified_dgs_data.csv", "data/stratified_pop_data.csv", "scripts/infer_ifr.R"]:
        get(base + f, "hopkins_ifr_gva/" + f.split("/")[-1])


# ---------------------------------------------------------------- England
UKARCH = "https://archive.ukhsa-dashboard.data.gov.uk/coronavirus-dashboard/"
ONS_CIS = ("https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/healthandsocialcare/conditionsanddiseases/"
           "datasets/coronaviruscovid19infectionsurveydata/2020/previous/")


def uk():
    for z, members in [("healthcare.zip", ["Healthcare/nhsRegion_newAdmissions.csv", "Healthcare/nation_newAdmissions.csv"]),
                       ("deaths.zip", ["Deaths/region_newDailyNsoDeathsByDeathDate.csv",
                                       "Deaths/region_newDeaths28DaysByDeathDate.csv",
                                       "Deaths/nation_newDailyNsoDeathsByDeathDate.csv",
                                       "Deaths/nation_newDeaths28DaysByDeathDate.csv"])]:
        p = get(UKARCH + z, "ukarchive/" + z)
        with zipfile.ZipFile(p) as zf:
            for m in members:
                if not (RAW / "ukarchive" / m).exists():
                    zf.extract(m, RAW / "ukarchive")
    get(ONS_CIS + "v20/covid19infectionsurveydatasets20200828.xlsx", "ons_cis/cis_covid19infectionsurveydatasets20200828.xlsx")
    get(ONS_CIS + "v27/covid19infectionsurveydatasets20201009.xlsx", "ons_cis/cis_covid19infectionsurveydatasets20201009.xlsx")
    # Populations. Original ONS mid-2019 release (LADs, regions; needs xlrd to read) and mid-2019 CCG estimates
    # carrying NHS England region codes; plus Nomis mid-2019/2020 regions (current, census-2021-rebased vintage).
    POP = "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/"
    get(POP + "populationestimatesforukenglandandwalesscotlandandnorthernireland/mid2019april2020localauthoritydistrictcodes/"
        "ukmidyearestimates20192020ladcodes.xls", "ons_pop/ukmidyearestimates20192020ladcodes.xls")
    z = get(POP + "clinicalcommissioninggroupmidyearpopulationestimates/mid2019sape22dt6a/"
            "sape22dt6amid2019ccg2020estimatesunformatted.zip", "ons_pop/sape22dt6a_mid2019_ccg2020.zip")
    with zipfile.ZipFile(z) as zf:
        for m in zf.namelist():
            if not (RAW / "ons_pop" / m).exists():
                zf.extract(m, RAW / "ons_pop")
    get("https://www.nomisweb.co.uk/api/v01/dataset/NM_2002_1.data.csv?geography=TYPE480&date=2019,2020&gender=0"
        "&c_age=200&measures=20100&select=date_name,geography_name,geography_code,obs_value", "ons_pop/nomis_regions_2019_2020.csv")


# ---------------------------------------------------------------- Papers (full text + supplements)
PMC = {"PMC7289564": "Stringhini et al. Lancet 2020 (SEROCoV-POP)",
       "PMC7876103": "Ward et al. Nat Commun 2021 (REACT-2)",
       "PMC7857406": "Buss et al. Science 2021 (Manaus)",
       "PMC7833057": "Perez-Saez et al. Lancet Infect Dis 2021 (Geneva IFR)"}


def papers():
    for pmc in PMC:
        n = pmc[3:]
        get("https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:"
            f"{n}&metadataPrefix=pmc", f"{pmc}.xml")
    for pmc in ["PMC7289564", "PMC7876103", "PMC7857406"]:
        p = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/supplementaryFiles", f"supp/{pmc}.zip")
        if not (RAW / "supp" / pmc).exists():
            with zipfile.ZipFile(p) as zf:
                zf.extractall(RAW / "supp" / pmc)


# ---------------------------------------------------------------- Brazil
def brazil():
    get("https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time_2020.csv.gz",
        "cases-brazil-cities-time_2020.csv.gz")
    get("https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv", "cases-brazil-states.csv")
    get("https://servicodados.ibge.gov.br/api/v3/agregados/6579/periodos/2019%7C2020/variaveis/9324"
        "?localidades=N6%5B1302603%5D%7CN3%5B13%5D", "ibge_pop_manaus_amazonas_2019_2020.json")


SIM = "https://diaad.s3.sa-east-1.amazonaws.com/sim/Mortalidade_Geral_{y}.csv"  # DATASUS / OpenDataSUS SIM
SIM_KEEP = ["DTOBITO", "CODMUNRES", "CODMUNOCOR", "CAUSABAS", "IDADE", "SEXO", "LOCOCOR", "TIPOBITO"]


def sim(years=range(2015, 2021)):
    """Stream the national SIM death-certificate files and keep only deaths with residence or occurrence in
    Amazonas (IBGE municipality codes starting 13). Full national files are not stored."""
    (RAW / "sim").mkdir(exist_ok=True)
    for y in years:
        dest = RAW / "sim" / f"sim_AM_{y}.csv"
        if dest.exists():
            print("have", dest.relative_to(HERE)); continue
        req = urllib.request.Request(SIM.format(y=y), headers=UA)
        n_all = n_keep = 0
        with urllib.request.urlopen(req, timeout=1800) as r, open(dest.with_suffix(".part"), "w", newline="") as f:
            rd = csv.DictReader(io.TextIOWrapper(r, encoding="latin-1"), delimiter=";")
            w = csv.writer(f); w.writerow(SIM_KEEP)
            for row in rd:
                n_all += 1
                if str(row.get("CODMUNRES", "")).startswith("13") or str(row.get("CODMUNOCOR", "")).startswith("13"):
                    w.writerow([row.get(k, "") for k in SIM_KEEP]); n_keep += 1
        dest.with_suffix(".part").rename(dest)
        print(f"sim {y}: {n_all} national records, {n_keep} kept (Amazonas residence or occurrence)")


# ---------------------------------------------------------------- OWID
def owid():
    get("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
        "owid-covid-data.csv")
    # The main OWID file now carries WHO weekly totals (daily rows zero except Sundays); OWID's archived JHU daily
    # series (deaths by report date) is fetched as well.
    get("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/full_data.csv", "owid_jhu_full_data.csv")


PARTS = {"swiss": swiss, "uk": uk, "papers": papers, "brazil": brazil, "sim": sim, "owid": owid}
if __name__ == "__main__":
    RAW.mkdir(exist_ok=True)
    for p in (sys.argv[1:] or list(PARTS)):
        PARTS[p]()
