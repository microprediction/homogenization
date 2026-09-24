"""Build the Spain first-wave (spring 2020) province dataset from primary sources.

Downloads (skipped if already present; pass --refresh to re-download):
  * ENE-COVID final report (Ministerio de Sanidad / ISCIII, July 2020), Tabla 4 page 15 -> seroprevalence by province
  * ISCIII/CNE casos_hosp_uci_def_sexo_edad_provres.csv -> daily cases/hospitalisations/ICU/deaths by province of residence
  * INE table 2852 (Padron, 1 Jan 2020) -> population by province
  * INE EDeS table 35176 -> weekly all-cause deaths by province (for excess mortality)
Writes spain_provinces.csv, spain_sero_all_rounds.csv, spain_daily.csv, spain_weekly_allcause.csv next to this file.
No number is typed by hand: the only hand-written tables are code/name mappings (ISO 3166-2:ES <-> INE code).
"""
import io, re, subprocess, sys, unicodedata, pathlib, urllib.request
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
URLS = {
    "ESTUDIO_ENE-COVID19_INFORME_FINAL.pdf": "https://www.sanidad.gob.es/areas/alertasEmergenciasSanitarias/alertasActuales/nCov/ene-covid/docs/ESTUDIO_ENE-COVID19_INFORME_FINAL.pdf",
    "ESTUDIO_ENE-COVID19_SEGUNDA_RONDA_INFORME_PRELIMINAR.pdf": "https://www.sanidad.gob.es/areas/alertasEmergenciasSanitarias/alertasActuales/nCov/ene-covid/docs/ESTUDIO_ENE-COVID19_SEGUNDA_RONDA_INFORME_PRELIMINAR.pdf",
    "ENE-COVID_CUARTA_RONDA.pdf": "https://www.sanidad.gob.es/gabinetePrensa/notaPrensa/pdf/15.12151220163348113.pdf",
    "ene_covid19_final_17.pdf": "https://portalcne.isciii.es/enecovid19/informes/ene_covid19_final_17.pdf",
    "casos_hosp_uci_def_sexo_edad_provres.csv": "https://cnecovid.isciii.es/covid19/resources/casos_hosp_uci_def_sexo_edad_provres.csv",
    "metadata_casos_hosp_uci_def_sexo_edad_provres.pdf": "https://cnecovid.isciii.es/covid19/resources/metadata_casos_hosp_uci_def_sexo_edad_provres.pdf",
    "ine_2852.csv": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/2852.csv",
    "ine_edes_35176.csv": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/35176.csv",
}
SERO_PDF, SERO_PAGE = "ESTUDIO_ENE-COVID19_INFORME_FINAL.pdf", 15   # Tabla 4 (test rapido), PDF page 15 of 32
ROUNDS = {1: ("2020-04-27", "2020-05-11"), 2: ("2020-05-18", "2020-06-01"), 3: ("2020-06-08", "2020-06-22")}  # report p.4 ("27 de abril a 11 de mayo ...")
DAILY_START, DAILY_END = "2020-02-01", "2020-12-31"
WAVE1_END = "2020-07-31"
# ENE-COVID round 4 (report of 15 Dec 2020, fieldwork 16-29 Nov 2020, report p.1). Province tables: (measure, table title, PDF page)
R4_PDF, R4_DATES = "ENE-COVID_CUARTA_RONDA.pdf", ("2020-11-16", "2020-11-29")
R4_TABLES = [
    ("current_r4", "Tabla 4. Prevalencia actual de anticuerpos IgG anti SARS-CoV-2 por provincia, Ronda 4", 13),
    ("global_r1_to_r4_r4participants", "Tabla 6. Prevalencia global (Rondas 1-4) de anticuerpos IgG anti SARS-CoV-2 por provincia", 16),
    ("seroconversion_r4_among_seronegative_r1_r3", "Tabla 12: Incidencia de seroconversión (nuevos IgG+) en personas seronegativas en las", 26),
    ("global_r1_to_r4_whole_cohort", "Tabla 4b. Prevalencia global (Rondas 1-4) de anticuerpos IgG anti SARS-CoV-2 por", 36),
]
METHOD = "IgG point-of-care lateral-flow immunochromatography (test rapido, Orient Gene Biotech COVID-19 IgG/IgM)"

# INE province code -> ISO 3166-2:ES subdivision code (as used in the CNE file)
INE2ISO = {"01": "VI", "02": "AB", "03": "A", "04": "AL", "05": "AV", "06": "BA", "07": "PM", "08": "B", "09": "BU",
           "10": "CC", "11": "CA", "12": "CS", "13": "CR", "14": "CO", "15": "C", "16": "CU", "17": "GI", "18": "GR",
           "19": "GU", "20": "SS", "21": "H", "22": "HU", "23": "J", "24": "LE", "25": "L", "26": "LO", "27": "LU",
           "28": "M", "29": "MA", "30": "MU", "31": "NA", "32": "OR", "33": "O", "34": "P", "35": "GC", "36": "PO",
           "37": "SA", "38": "TF", "39": "S", "40": "SG", "41": "SE", "42": "SO", "43": "T", "44": "TE", "45": "TO",
           "46": "V", "47": "VA", "48": "BI", "49": "ZA", "50": "Z", "51": "CE", "52": "ML"}
# ENE-COVID Tabla 4 names that do not match the INE name after normalisation
ENE_NAME_FIX = {"principado asturias": "asturias", "santa cruz tenerife": "santa cruz de tenerife"}


def fetch(refresh=False):
    for name, url in URLS.items():
        p = HERE / name
        if refresh or not p.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            p.write_bytes(urllib.request.urlopen(req, timeout=300).read())
        print(f"{name}: {p.stat().st_size} bytes")


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()
    return ENE_NAME_FIX.get(s, s)


def num(s):  # Spanish number: '.' thousands, ',' decimal
    return float(s.replace(".", "").replace(",", "."))


def population():
    d = pd.read_csv(HERE / "ine_2852.csv", sep=";", dtype=str, encoding="utf-8-sig")
    d = d[(d["Sexo"] == "Total") & (d["Periodo"] == "2020") & (d["Provincias"] != "Total")].copy()
    d["ine_code"] = d["Provincias"].str[:2]
    d["name"] = d["Provincias"].str[3:]
    d["population_2020"] = d["Total"].str.replace(".", "", regex=False).astype(int)
    d["province_code"] = d["ine_code"].map(INE2ISO)
    assert d["province_code"].notna().all() and len(d) == 52
    return d[["province_code", "ine_code", "name", "population_2020"]].sort_values("ine_code").reset_index(drop=True)


def province_table(pdf, page, title, pop):
    """Parse an ENE-COVID province table with 3 column groups of (N, %, CI low - CI high) from one PDF page.
    Returns {province_code: [(n, pct, lo, hi) x 3]} for the 52 areas."""
    txt = subprocess.run(["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(HERE / pdf), "-"],
                         capture_output=True, text=True, check=True).stdout
    assert title in txt, (pdf, page, title)
    cell = r"(\d+)\s+(\d+,\d)\s+(\d+,\d)\s*-?\s*(\d+,\d)"
    rx = re.compile(r"^\s*(\D+?)\s+" + r"\s+".join([cell] * 3) + r"\s*$")
    name2code = {norm(n): c for n, c in zip(pop["name"], pop["province_code"])}
    out = {}
    for line in txt.split(title, 1)[1].splitlines():
        m = rx.match(line)
        if m:
            g = m.groups()
            out[name2code[norm(g[0])]] = [(int(g[i]), num(g[i + 1]), num(g[i + 2]), num(g[i + 3])) for i in (1, 5, 9)]
    assert len(out) == 52, (pdf, page, len(out))
    return out


def sero(pop):
    t = province_table(SERO_PDF, SERO_PAGE, "Tabla 4: Prevalencia de anticuerpos IgG anti SARS-CoV-2 por provincia (test rápido)", pop)
    rows = []
    for code, cells in t.items():
        for r in (1, 2, 3):
            n, p, lo, hi = cells[r - 1]
            rows.append(dict(province_code=code, round=r, round_start=ROUNDS[r][0], round_end=ROUNDS[r][1],
                             n_participants=n, sero_pct=p, ci_low_pct=lo, ci_high_pct=hi, sero_method=METHOD,
                             source=f"{SERO_PDF} Tabla 4, PDF page {SERO_PAGE}"))
    s = pd.DataFrame(rows)
    assert s["province_code"].nunique() == 52, s["province_code"].nunique()
    s["note"] = ""
    s.loc[(s["province_code"] == "CE") & (s["round"] == 2), "note"] = \
        "Tabla 4 prints CI 0,1-8,8; Tabla 3 (p.11, CCAA table, Ceuta) prints 0,1-1,7 for the same estimate; value kept as printed in Tabla 4"
    return s


def sero_round4(pop):
    """ENE-COVID round 4: all four province tables, 'Total' column group only (the other two are men / women)."""
    rows = []
    for measure, title, page in R4_TABLES:
        for code, cells in province_table(R4_PDF, page, title, pop).items():
            n, p, lo, hi = cells[0]
            rows.append(dict(province_code=code, measure=measure, round_start=R4_DATES[0], round_end=R4_DATES[1],
                             n_participants=n, pct=p, ci_low_pct=lo, ci_high_pct=hi, sero_method=METHOD,
                             source=f"{R4_PDF} {title.split('.')[0].split(':')[0]}, PDF page {page}"))
    return pd.DataFrame(rows)


def daily():
    d = pd.read_csv(HERE / "casos_hosp_uci_def_sexo_edad_provres.csv", keep_default_na=False)  # 'NC' = no consta
    d = d[(d["fecha"] >= DAILY_START) & (d["fecha"] <= DAILY_END)]
    g = d.groupby(["fecha", "provincia_iso"], as_index=False)[["num_def", "num_hosp", "num_uci", "num_casos"]].sum()
    g.columns = ["date", "province_code", "deaths", "hosp_admissions", "icu_admissions", "cases"]
    iso2ine = {v: k for k, v in INE2ISO.items()}
    g.insert(2, "ine_code", g["province_code"].map(iso2ine).fillna(""))  # '' for NC (province of residence not recorded)
    return g.sort_values(["province_code", "date"]).reset_index(drop=True)


def weekly_allcause():
    d = pd.read_csv(HERE / "ine_edes_35176.csv", sep=";", dtype=str, encoding="utf-8-sig")
    d = d[(d["Tipo de dato"] == "Dato base") & d["Provincias"].notna() & (d["Total Nacional"] == "Total Nacional")].copy()
    d["year"] = d["Periodo"].str[:4].astype(int)
    d["week"] = d["Periodo"].str[6:].astype(int)
    d = d[(d["year"] >= 2015) & (d["year"] <= 2020)]
    d["ine_code"] = d["Provincias"].str[:2]
    d["province_code"] = d["ine_code"].map(INE2ISO)
    d["deaths_allcause"] = pd.to_numeric(d["Total"].str.replace(".", "", regex=False), errors="coerce")
    w = d.pivot_table(index=["province_code", "ine_code", "week"], columns="year", values="deaths_allcause").reset_index()
    base = [y for y in range(2015, 2020)]
    w["mean_2015_2019"] = w[base].mean(axis=1).round(1)  # derived: simple same-week mean, NOT an official excess estimate
    w["excess_2020_vs_mean_2015_2019"] = (w[2020] - w["mean_2015_2019"]).round(1)
    w.columns = [f"deaths_{c}" if isinstance(c, int) else c for c in w.columns]
    return w.sort_values(["province_code", "week"]).reset_index(drop=True)


if __name__ == "__main__":
    fetch("--refresh" in sys.argv)
    pop = population()
    s = sero(pop)
    s.to_csv(HERE / "spain_sero_all_rounds.csv", index=False)
    r3 = s[s["round"] == 3].rename(columns={"sero_pct": "sero_prevalence_pct", "n_participants": "sero_n_participants"})
    r3 = r3.assign(sero_round="ENE-COVID round 3 (2020-06-08 to 2020-06-22)")
    prov = pop.merge(r3[["province_code", "sero_prevalence_pct", "ci_low_pct", "ci_high_pct", "sero_round", "sero_method",
                         "sero_n_participants", "source", "note"]], on="province_code", how="left")
    prov = prov.rename(columns={"source": "sero_source", "note": "sero_note"})
    dd = daily()
    dd.to_csv(HERE / "spain_daily.csv", index=False)
    w1 = dd[dd["date"] <= WAVE1_END]
    tot = w1.groupby("province_code")[["deaths", "hosp_admissions"]].sum().add_suffix(f"_cne_{DAILY_START}_{WAVE1_END}")
    prov = prov.merge(tot, left_on="province_code", right_index=True, how="left")
    r4 = sero_round4(pop)
    r4.to_csv(HERE / "spain_sero_round4.csv", index=False)
    short = {"current_r4": "sero_r4", "seroconversion_r4_among_seronegative_r1_r3": "seroconv_r4",
             "global_r1_to_r4_r4participants": "sero_global_r1_r4"}
    for measure, pre in short.items():
        x = r4[r4["measure"] == measure].set_index("province_code")[["pct", "ci_low_pct", "ci_high_pct", "n_participants"]]
        x.columns = [f"{pre}_pct", f"{pre}_ci_low_pct", f"{pre}_ci_high_pct", f"{pre}_n"]
        prov = prov.merge(x, left_on="province_code", right_index=True, how="left")
    prov.to_csv(HERE / "spain_provinces.csv", index=False)
    weekly_allcause().to_csv(HERE / "spain_weekly_allcause.csv", index=False)
    print(prov[["province_code", "name", "population_2020", "sero_prevalence_pct", "ci_low_pct", "ci_high_pct"]].to_string())
    print("daily rows", len(dd), dd["date"].min(), dd["date"].max(), "provinces", dd["province_code"].nunique())
