"""Build tidy CSVs for the three point checks from the files fetch_points.py put in raw/. Every number comes from a
downloaded file (CSV, XLSX, JATS XML, or text extracted with pdftotext from a supplementary PDF); nothing is typed in.
Outputs (this directory): geneva_daily.csv, geneva_serology.csv, england_daily.csv, england_serology.csv,
manaus_daily.csv, manaus_sim_allcause_by_year.csv, manaus_serology.csv, populations.csv, owid_subset.csv.
Needs pandas, openpyxl, pdftotext (poppler); xlrd optional (original ONS mid-2019 .xls)."""
import html, json, pathlib, re, subprocess
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
START, END = "2020-01-01", "2020-12-31"
POPS = []  # rows for populations.csv


def pct(s):
    """'4·8' or '11. 9' -> 4.8 / 11.9 (source typography: middle dots, stray spaces)."""
    return float(s.replace("·", ".").replace(" ", ""))


def jats_tables(xml_path):
    x = xml_path.read_text()
    out = {}
    for tw in re.findall(r"<table-wrap.*?</table-wrap>", x, re.S):
        lab = re.search(r"<label>(.*?)</label>", tw).group(1)
        rows = []
        for tr in re.findall(r"<tr.*?</tr>", tw, re.S):
            rows.append([html.unescape(re.sub(r"\s+", " ", re.sub("<[^>]+>", "", c))).strip()
                         for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)])
        out[lab] = rows
    return out


def pdf_text(pdf):
    return subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout


def daterange_frame():
    return pd.DataFrame({"date": pd.date_range(START, END).strftime("%Y-%m-%d")})


# =============================================================== Geneva
def geneva():
    d = daterange_frame()
    foph_d = pd.read_csv(RAW / "foph_COVID19Death_geoRegion.csv")
    foph_h = pd.read_csv(RAW / "foph_COVID19Hosp_geoRegion.csv")
    g_d = foph_d[foph_d.geoRegion == "GE"][["datum", "entries"]].rename(columns={"datum": "date", "entries": "deaths"})
    g_h = foph_h[foph_h.geoRegion == "GE"][["datum", "entries"]].rename(columns={"datum": "date", "entries": "hosp_admissions"})
    d = d.merge(g_d, on="date", how="left").merge(g_h, on="date", how="left")
    # openZH cantonal series (as reported by the canton, cumulative deaths and new hospitalisations by report date)
    oz = pd.read_csv(RAW / "COVID19_Fallzahlen_CH_total_v2.csv")
    oz = oz[oz.abbreviation_canton_and_fl == "GE"].sort_values("date").drop_duplicates("date", keep="last")
    oz = oz[["date", "ncumul_deceased", "new_hosp"]].rename(
        columns={"ncumul_deceased": "deaths_cumulative_canton", "new_hosp": "hosp_new_canton"})
    d = d.merge(oz, on="date", how="left")
    d["deaths_canton_daily_diff"] = d.deaths_cumulative_canton.diff()
    # Perez-Saez et al. (2021) study data: deaths by age class, split by care-home (EMS) residence
    hk = pd.read_csv(RAW / "hopkins_ifr_gva" / "stratified_dgs_data.csv").dropna(subset=["date"])
    hk = hk[hk["var"] == "death_incid"]
    tot = hk.groupby("date").value.sum().rename("deaths_ifr_study").reset_index()
    ems = hk[hk.EMS == 1].groupby("date").value.sum().rename("deaths_ifr_study_care_homes").reset_index()
    d = d.merge(tot, on="date", how="left").merge(ems, on="date", how="left")
    d.insert(1, "place", "Geneva")
    cols = ["date", "place", "deaths", "hosp_admissions", "deaths_cumulative_canton", "deaths_canton_daily_diff",
            "hosp_new_canton", "deaths_ifr_study", "deaths_ifr_study_care_homes"]
    d[cols].to_csv(HERE / "geneva_daily.csv", index=False)

    POPS.append(dict(place="Geneva", geography="canton", population=int(foph_d[foph_d.geoRegion == "GE"]["pop"].iloc[0]),
                     year="FOPH 'pop' column (reference year not stated in file)", source="raw/foph_COVID19Death_geoRegion.csv"))
    hp = pd.read_csv(RAW / "hopkins_ifr_gva" / "stratified_pop_data.csv")
    POPS.append(dict(place="Geneva", geography="canton", population=int(hp["pop"].sum()),
                     year="as used by Perez-Saez et al. 2021 (sum of age classes)", source="raw/hopkins_ifr_gva/stratified_pop_data.csv"))

    # ---- serology
    rows = []
    t2 = jats_tables(RAW / "PMC7289564.xml")["Table 2"]
    ref = re.findall(r'"(2020-\d\d-\d\d)", ([\d.]+),\s*([\d.]+),\s*([\d.]+)',
                     (RAW / "hopkins_ifr_gva" / "infer_ifr.R").read_text())
    src_main = "Stringhini et al. Lancet 2020;396:313-19, doi:10.1016/S0140-6736(20)31304-0, Table 2 (PMC7289564 XML)"
    for r in t2:
        m = re.match(r"Week (\d) \(n=(\d+)\)", r[0])
        if not m:
            continue
        wk, n = int(m.group(1)), int(m.group(2))
        npos, rawp = re.match(r"(\d+) \(([\d·]+)%\)", r[1]).groups()
        est, lo, hi = re.match(r"([\d·]+)% \(([\d·]+)–([\d·]+)\)", r[4]).groups()
        start = pd.Timestamp("2020-04-06") + pd.Timedelta(days=7 * (wk - 1))
        common = dict(place="Geneva", week=wk, date_start=start.strftime("%Y-%m-%d"),
                      date_end=min(start + pd.Timedelta(days=6), pd.Timestamp("2020-05-09")).strftime("%Y-%m-%d"),
                      date_basis="week boundaries inferred (paper gives only 6 Apr-9 May 2020 for weeks 1-5)",
                      ref_date=ref[wk - 1][0] if len(ref) >= wk else "", n_tested=n, n_positive=int(npos))
        rows.append(dict(common, prevalence_pct=pct(est), ci_low_pct=pct(lo), ci_high_pct=pct(hi),
                         method="IgG ELISA (Euroimmun S1, cutoff 1.1) with rIFA confirmation; Bayesian logistic model, "
                                "household random effect, test sensitivity/specificity, post-stratified to Geneva age/sex",
                         adjusted="adjusted", sample="SEROCoV-POP: Bus Sante participants + household members (age>=5)",
                         source=src_main))
        rows.append(dict(common, prevalence_pct=pct(rawp), ci_low_pct=None, ci_high_pct=None,
                         method="share of tested participants ELISA-positive (no CI in source)", adjusted="raw",
                         sample="SEROCoV-POP: Bus Sante participants + household members (age>=5)", source=src_main))
    txt = pdf_text(RAW / "supp" / "PMC7289564" / "mmc1.pdf")
    for tab, desc, sample in [
            ("Table S2", "as main model, Bus Sante participants only", "Bus Sante participants only"),
            ("Table S4", "as main model, ELISA positivity cutoff 1.5 (Meyer et al.)", "Bus Sante + household"),
            ("Table S5", "as main model, ELISA <0.5 negative, rIFA result for all others", "Bus Sante + household")]:
        block = txt[txt.index(tab + ":"):]
        for wk in range(1, 6):
            m = re.search(rf"^\s+{wk}\s+(\d+)\s+(\d+) \(([\d.]+)%\).*?([\d.]+) \(([\d.]+)-([\d.]+)\)", block, re.M)
            n, npos, rawp, est, lo, hi = m.groups()
            start = pd.Timestamp("2020-04-06") + pd.Timedelta(days=7 * (wk - 1))
            rows.append(dict(place="Geneva", week=wk, date_start=start.strftime("%Y-%m-%d"),
                             date_end=min(start + pd.Timedelta(days=6), pd.Timestamp("2020-05-09")).strftime("%Y-%m-%d"),
                             date_basis="week boundaries inferred (paper gives only 6 Apr-9 May 2020 for weeks 1-5)",
                             ref_date=ref[wk - 1][0] if len(ref) >= wk else "", n_tested=int(n), n_positive=int(npos),
                             prevalence_pct=float(est), ci_low_pct=float(lo), ci_high_pct=float(hi),
                             method=desc, adjusted="adjusted", sample=sample,
                             source=f"Stringhini et al. 2020 appendix (mmc1.pdf), {tab}"))
    pd.DataFrame(rows).to_csv(HERE / "geneva_serology.csv", index=False)


# =============================================================== England
def england():
    uk = RAW / "ukarchive"
    d = daterange_frame()

    def series(path, name):
        s = pd.read_csv(path)
        return s[["date", "area_type", "area_name", "value"]].rename(columns={"value": name, "area_name": "place"})
    ons = series(uk / "Deaths/region_newDailyNsoDeathsByDeathDate.csv", "deaths")
    d28 = series(uk / "Deaths/region_newDeaths28DaysByDeathDate.csv", "deaths_28d_positive_test")
    reg = ons.merge(d28, on=["date", "area_type", "place"], how="outer")
    nat = series(uk / "Deaths/nation_newDailyNsoDeathsByDeathDate.csv", "deaths").merge(
        series(uk / "Deaths/nation_newDeaths28DaysByDeathDate.csv", "deaths_28d_positive_test"),
        on=["date", "area_type", "place"], how="outer").merge(
        series(uk / "Healthcare/nation_newAdmissions.csv", "hosp_admissions"), on=["date", "area_type", "place"], how="outer")
    nat = nat[nat.place == "England"]
    nhs = series(uk / "Healthcare/nhsRegion_newAdmissions.csv", "hosp_admissions")
    out = pd.concat([nat, reg, nhs], ignore_index=True)
    out["geography"] = out.area_type.map({"nation": "nation", "region": "ONS region", "nhsRegion": "NHS England region"})
    out = out[(out.date >= START) & (out.date <= END)].sort_values(["geography", "place", "date"])
    out[["date", "place", "geography", "deaths", "deaths_28d_positive_test", "hosp_admissions"]].to_csv(
        HERE / "england_daily.csv", index=False)

    # populations: NHS England regions = sum of mid-2019 CCG estimates by NHSER20 code
    ccg = pd.read_excel(RAW / "ons_pop" / "SAPE22DT6a-mid-2019-ccg-2020-estimates-unformatted.xlsx",
                        sheet_name="Mid-2019 Persons", header=6).dropna(subset=["CCG Code"])
    for name, v in ccg.groupby("NHSER20 Name")["All Ages"].sum().items():
        POPS.append(dict(place=name, geography="NHS England region", population=int(v), year="mid-2019",
                         source="ONS SAPE22DT6a mid-2019 CCG (2020 boundaries), summed by NHSER20"))
    try:
        mye = pd.read_excel(RAW / "ons_pop" / "ukmidyearestimates20192020ladcodes.xls", sheet_name="MYE2 - Persons",
                            header=None)
        for _, r in mye[mye[2].isin(["Region", "Country"])].iterrows():
            POPS.append(dict(place=str(r[1]).title(), geography="ONS region" if r[2] == "Region" else "country",
                             population=int(r[3]), year="mid-2019 (original June 2020 release)",
                             source="ONS ukmidyearestimates20192020ladcodes.xls, MYE2 - Persons"))
    except ImportError:
        print("xlrd missing: skipping original ONS mid-2019 .xls (Nomis series still used)")
    nom = pd.read_csv(RAW / "ons_pop" / "nomis_regions_2019_2020.csv")
    for _, r in nom.iterrows():
        POPS.append(dict(place=r.GEOGRAPHY_NAME, geography="ONS region" if r.GEOGRAPHY_CODE.startswith("E12") else "country",
                         population=int(r.OBS_VALUE), year=f"mid-{r.DATE_NAME} (Nomis NM_2002_1, current vintage)",
                         source="raw/ons_pop/nomis_regions_2019_2020.csv"))

    # ---- serology
    rows = []
    txt = (RAW / "supp" / "PMC7876103" / "41467_2021_21237_MOESM1_ESM.pdf")
    txt = pdf_text(txt)
    block = txt[re.search(r"^\f?Supplementary Table 1:", txt, re.M).start():re.search(r"^\f?Supplementary Table 2:", txt, re.M).start()]
    block = block[block.index("\nEngland "):]
    src = ("Ward et al. Nat Commun 2021;12:905, doi:10.1038/s41467-021-21237-w, Supplementary Table 1 "
           "(MOESM1_ESM.pdf); REACT-2 round 1, LFIA self-test, adults 18+")
    names = {"Yorkshire": "Yorkshire and The Humber"}
    region_lines = block[block.index("Region"):] if "Region\n" in block else block
    wanted = ["England", "North East", "North West", "Yorkshire", "East Midlands", "West Midlands",
              "East of England", "London", "South East", "South West"]
    for w in wanted:
        src_block = block if w == "England" else region_lines
        m = re.search(rf"^{w}\s+(\d+)\s+(\d+)\s+([\d.]+) \[([\d.]+)-([\d.]+)\]\s+([\d.]+) \[([\d.]+)-([\d.]+)\]\s+"
                      rf"([\d.]+) \[([\d.]+)-([\d.]+)\]", src_block, re.M)
        n, npos, *v = m.groups()
        v = list(map(float, v))
        for (p, lo, hi), how, method in zip([v[0:3], v[3:6], v[6:9]], ["raw", "adjusted", "adjusted"],
                                            ["crude prevalence", "adjusted for test sensitivity/specificity",
                                             "adjusted for test sensitivity/specificity and re-weighted for sample design/response"]):
            rows.append(dict(place=names.get(w, w), geography="nation" if w == "England" else "ONS region",
                             date_start="2020-06-20", date_end="2020-07-13", n_tested=int(n), n_positive=int(npos),
                             prevalence_pct=p, ci_low_pct=lo, ci_high_pct=hi, method="REACT-2 IgG LFIA, " + method,
                             adjusted=how, source=src))
    for f, end, rel in [("cis_covid19infectionsurveydatasets20200828.xlsx", "2020-08-09", "28 Aug 2020"),
                        ("cis_covid19infectionsurveydatasets20201009.xlsx", "2020-09-08", "9 Oct 2020")]:
        x = pd.read_excel(RAW / "ons_cis" / f, sheet_name="3b", header=None)
        title = str(x.iloc[2, 0])
        weighted = "(weighted)" in title
        for _, r in x.iloc[6:15].iterrows():
            rows.append(dict(place=str(r[0]).replace("the Humber", "The Humber"), geography="ONS region",
                             date_start="2020-04-26", date_end=end, n_tested=int(r[5]), n_positive=int(r[4]),
                             prevalence_pct=round(100 * r[1], 2), ci_low_pct=round(100 * r[2], 2),
                             ci_high_pct=round(100 * r[3], 2),
                             method=f"ONS COVID-19 Infection Survey, ever antibody-positive, age 16+, Bayesian model; "
                                    f"sheet title: '{title.strip()}'",
                             adjusted="adjusted (weighted)" if weighted else "modelled, unweighted per sheet title",
                             source=f"ONS CIS datasets release {rel}, Table 3b (raw/ons_cis/{f})"))
        e = pd.read_excel(RAW / "ons_cis" / f, sheet_name="3a", header=None)
        rows.append(dict(place="England", geography="nation", date_start="2020-04-26", date_end=end,
                         n_tested=int(e.iloc[6, 1]), n_positive=int(e.iloc[6, 2]),
                         prevalence_pct=round(100 * e.iloc[7, 2], 2), ci_low_pct=round(100 * e.iloc[7, 3], 2),
                         ci_high_pct=round(100 * e.iloc[7, 4], 2),
                         method=f"ONS COVID-19 Infection Survey, ever antibody-positive, age 16+; sheet title: "
                                f"'{str(e.iloc[2, 0]).strip()}'", adjusted="adjusted (weighted)",
                         source=f"ONS CIS datasets release {rel}, Table 3a (raw/ons_cis/{f})"))
    pd.DataFrame(rows).to_csv(HERE / "england_serology.csv", index=False)


# =============================================================== Manaus
def manaus():
    d = daterange_frame()
    c = pd.read_csv(RAW / "cases-brazil-cities-time_2020.csv.gz")
    m = c[c.ibgeID == 1302603][["date", "newDeaths", "deaths"]].rename(
        columns={"newDeaths": "deaths_covid_reported", "deaths": "deaths_covid_reported_cumulative"})
    s = pd.read_csv(RAW / "cases-brazil-states.csv")
    s = s[s.state == "AM"][["date", "newDeaths", "deaths"]].rename(
        columns={"newDeaths": "deaths_covid_reported", "deaths": "deaths_covid_reported_cumulative"})
    sims = {}
    for y in range(2015, 2021):
        f = RAW / "sim" / f"sim_AM_{y}.csv"
        if f.exists():
            sims[y] = pd.read_csv(f, dtype=str)
    byyear = []
    for place, pred in [("Manaus", lambda t: t.CODMUNRES == "130260"), ("Amazonas", lambda t: t.CODMUNRES.str.startswith("13"))]:
        for y, t in sims.items():
            t = t[pred(t)].copy()
            t["date"] = pd.to_datetime(t.DTOBITO.str.zfill(8), format="%d%m%Y", errors="coerce").dt.strftime("%Y-%m-%d")
            a = t.groupby("date").size().rename("deaths_all_cause").reset_index()
            b = t[t.CAUSABAS == "B342"].groupby("date").size().rename("deaths_underlying_B342").reset_index()
            byyear.append(a.merge(b, on="date", how="left").fillna({"deaths_underlying_B342": 0}).assign(place=place, year=y))
    byyear = pd.concat(byyear, ignore_index=True) if byyear else pd.DataFrame()
    if len(byyear):
        byyear[["place", "year", "date", "deaths_all_cause", "deaths_underlying_B342"]].sort_values(
            ["place", "date"]).to_csv(HERE / "manaus_sim_allcause_by_year.csv", index=False)
    out = []
    for place, rep in [("Manaus", m), ("Amazonas", s)]:
        x = d.merge(rep, on="date", how="left")
        if len(byyear):
            b = byyear[(byyear.place == place)]
            x = x.merge(b[b.year == 2020][["date", "deaths_all_cause", "deaths_underlying_B342"]], on="date", how="left")
            x["md"] = x.date.str[5:]
            for y in range(2015, 2020):
                prev = b[b.year == y].assign(md=lambda q: q.date.str[5:])[["md", "deaths_all_cause"]]
                x = x.merge(prev.rename(columns={"deaths_all_cause": f"deaths_all_cause_{y}_same_day"}), on="md", how="left")
            x = x.drop(columns="md")
        x.insert(1, "place", place)
        x["hosp_admissions"] = pd.NA
        out.append(x)
    pd.concat(out).to_csv(HERE / "manaus_daily.csv", index=False)

    ib = json.loads((RAW / "ibge_pop_manaus_amazonas_2019_2020.json").read_text())
    for r in ib[0]["resultados"]:
        for sr in r["series"]:
            for yr, v in sr["serie"].items():
                POPS.append(dict(place=sr["localidade"]["nome"], geography="municipality" if sr["localidade"]["nivel"]["id"] == "N6"
                                 else "state", population=int(v), year=f"{yr} (IBGE estimate, 1 July)",
                                 source="IBGE SIDRA API agregado 6579 variavel 9324"))

    # ---- serology: Buss et al. Science 2021, Table S2
    txt = pdf_text(RAW / "buss_science" / "abe9728_buss_sm.pdf")
    end = re.search(r"^Table S2\.", txt, re.M).start()
    blk = txt[txt.rfind("1.4 S/C threshold", 0, end):end]
    num = r"(\d+\.?\s?\d*)"
    ci = rf"{num}\s*\(\s*{num}\s*-\s*{num}\s*\)"
    rows = []
    city = None
    months = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
    for line in blk.splitlines():
        ls = line.strip()
        if ls in ("Manaus", "São Paulo"):
            city = ls; continue
        mm = re.match(r"(\w{3}) (\d+)\w*-(\d+)\w*\s+(\d+)\s+(\d+)\s+(.*)", ls)
        if not mm:
            continue
        mon, d0, d1, n, npos, rest = mm.groups()
        cis = re.findall(ci, rest)
        seror = None
        # layout: crude, weighted, sens/spec-adj, [seroreversion-adj or '-'], positives@0.4, crude, weighted, adj
        parts = re.split(r"\s{2,}", rest.strip())
        if parts[3].strip() == "-":
            cis_14, cis_04 = cis[:3], cis[3:]
            npos04 = parts[4]
        else:
            cis_14, seror, cis_04 = cis[:3], cis[3], cis[4:]
            npos04 = parts[4]
        base = dict(place=city, date_start=f"2020-{months[mon]:02d}-{int(d0):02d}", date_end=f"2020-{months[mon]:02d}-{int(d1):02d}",
                    source="Buss et al. Science 2021;371:288-292, doi:10.1126/science.abe9728, Table S2 (supplementary PDF)")
        labels = [("crude", "raw"), ("age-sex weighted", "adjusted"), ("weighted + sensitivity/specificity (Rogan-Gladen)", "adjusted")]
        for thr, trip, npp in [("1.4", cis_14, npos), ("0.4", cis_04, npos04)]:
            for (lab, how), (p, lo, hi) in zip(labels, trip):
                rows.append(dict(base, n_tested=int(n), n_positive=int(npp), prevalence_pct=pct(p), ci_low_pct=pct(lo),
                                 ci_high_pct=pct(hi), method=f"Abbott SARS-CoV-2 IgG CMIA, S/C threshold {thr}; {lab}",
                                 adjusted=how))
        if seror:
            p, lo, hi = seror
            rows.append(dict(base, n_tested=int(n), n_positive=int(npos), prevalence_pct=pct(p), ci_low_pct=pct(lo),
                             ci_high_pct=pct(hi), method="Abbott IgG, S/C 1.4; weighted, sens/spec-adjusted, and "
                                                         "corrected for seroreversion (cumulative attack rate)",
                             adjusted="adjusted (seroreversion)"))
    pd.DataFrame(rows).to_csv(HERE / "manaus_serology.csv", index=False)


def owid():
    cols = ["iso_code", "location", "date", "population", "new_deaths", "new_deaths_smoothed", "total_deaths",
            "new_deaths_smoothed_per_million", "stringency_index"]
    o = pd.read_csv(RAW / "owid-covid-data.csv", usecols=cols)
    keep = ["Switzerland", "United Kingdom", "Brazil", "Sweden", "Spain"]
    o = o[o.location.isin(keep)]
    j = pd.read_csv(RAW / "owid_jhu_full_data.csv", usecols=["date", "location", "new_deaths", "weekly_deaths"])
    j = j[j.location.isin(keep)].rename(columns={"new_deaths": "new_deaths_jhu_daily",
                                                 "weekly_deaths": "weekly_deaths_jhu_trailing7d"})
    o = o.rename(columns={"new_deaths": "new_deaths_who_weekly", "new_deaths_smoothed": "new_deaths_smoothed_who_weekly"})
    o = o.merge(j, on=["location", "date"], how="outer").sort_values(["location", "date"])
    o["iso_code"] = o.groupby("location").iso_code.transform("first")
    o.to_csv(HERE / "owid_subset.csv", index=False)


if __name__ == "__main__":
    geneva(); england(); manaus(); owid()
    pops = pd.DataFrame(POPS)
    fix = {"East": "East of England", "Yorkshire And The Humber": "Yorkshire and The Humber", "England And Wales": "England and Wales"}
    pops["place"] = pops.place.replace(fix)
    pops.to_csv(HERE / "populations.csv", index=False)
    for f in sorted(HERE.glob("*.csv")):
        print(f.name, sum(1 for _ in open(f)) - 1, "rows")
