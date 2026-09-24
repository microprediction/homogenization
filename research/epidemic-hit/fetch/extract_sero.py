#!/usr/bin/env python3
"""Extract Swedish 2020 (to Mar 2021) SARS-CoV-2 seroprevalence estimates from the
downloaded PDFs / supplementary files in raw/sero/ into sero_estimates.csv.

Every number is parsed from the source text (pdfplumber page text, JATS XML, or xlsx);
nothing is typed in by hand. Page numbers are found by locating the parsed string on a
PDF page. Rows are restricted to samples collected up to 2021-03-12 (before most adults
were vaccinated; vaccination started 27 Dec 2020 with care-home residents).

Run from data/sweden/:  python3 scripts/extract_sero.py
"""
import csv
import datetime as dt
import html
import re
from pathlib import Path

import openpyxl
import pdfplumber

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "raw" / "sero"
OUT = HERE / "sero_estimates.csv"

REGION = {  # SCB län codes
    "Stockholm": "01", "Stockholmsregionen": "01", "Uppsala": "03", "Jönköping": "06",
    "Kalmar": "08", "Skåne": "12", "Västra Götaland": "14", "Väster Götland": "14",
    "Västra": "14", "Örebro": "18", "Västerbotten": "24", "Jämtland": "23",
}
CANON = {"Stockholmsregionen": "Stockholm", "Väster Götland": "Västra Götaland", "Västra": "Västra Götaland"}

FHM_OPEN_URL = "https://www.folkhalsomyndigheten.se/contentassets/9c5893f84bd049e691562b9eeb0ca280/pavisning-antikroppar-mot-sars-cov-2-blodprov-oppenvarden.pdf"
FHM_OPEN_D1_URL = "https://www.folkhalsomyndigheten.se/contentassets/9c5893f84bd049e691562b9eeb0ca280/pavisning-antikroppar-genomgangen-covid-19-blodprov-oppenvarden-delrapport-1.pdf"
FHM_BD_URL = "https://www.folkhalsomyndigheten.se/contentassets/376f9021a4c84da08de18ac597284f0c/pavisning-antikroppar-mot-sars-cov-2-blodgivare.pdf"
FHM_BD_D2_URL = "https://www.folkhalsomyndigheten.se/contentassets/376f9021a4c84da08de18ac597284f0c/pavisning-antikroppar-genomgangen-covid-19-blodgivare-delrapport-2.pdf"
RK_URL = "https://www.folkhalsomyndigheten.se/contentassets/2cf102cd299c4382b9a0447dc0626356/forekomsten-antikroppar-rinkeby-kista.pdf"
NEWS_URL = "https://www.folkhalsomyndigheten.se/nyheter-och-press/nyhetsarkiv/2020/maj/forsta-resultaten-fran-pagaende-undersokning-av-antikroppar-for-covid-19-virus/"
CD_URL = "https://doi.org/10.1111/joim.13304 (PMC8242905; supplement via https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8242905/supplementaryFiles)"
RX_URL = "https://www.nature.com/articles/s41467-021-23893-4.pdf"

FHM_METHOD = "SciLifeLab/KTH multiplex bead IgG (>=2 of 3 antigens: S1S2 foldon, S1, N; spec 98.9%, sens 99.4%)"
FHM_ADJ_FINAL = "weighted by age group x region population (SCB) + Rogan-Gladen test-performance correction; Clopper-Pearson 95% CI"
FHM_ADJ_2020 = "age-weighted, Rogan-Gladen corrected; Clopper-Pearson 95% CI (preliminary 2020 method, R 3.6.2)"
FHM_BD_ADJ_2020 = "share positive, Rogan-Gladen corrected; Clopper-Pearson 95% CI (preliminary 2020 method)"
FHM_BD_ADJ_FINAL = "weighted (SCB population 20-64, 2020) + Rogan-Gladen correction; Clopper-Pearson 95% CI"

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "maj": 5, "jun": 6, "jul": 7, "aug": 8,
          "sep": 9, "okt": 10, "nov": 11, "dec": 12}
CUTOFF = dt.date(2021, 3, 12)

FIELDS = ["study_id", "region_code", "region_name", "sample_start", "sample_end", "iso_weeks",
          "period_as_printed", "population_sampled", "n_tested", "n_positive", "prevalence_pct",
          "ci_low_pct", "ci_high_pct", "adjustment", "method", "report_version", "source_file",
          "source_url", "table_or_figure", "page", "notes"]


def num(s):
    return None if s is None else float(s.replace(",", ".").strip())


def iso_bounds(year, w1, w2=None):
    w2 = w2 or w1
    return dt.date.fromisocalendar(year, w1, 1), dt.date.fromisocalendar(year, w2, 7)


class Pdf:
    def __init__(self, path):
        self.path = path
        with pdfplumber.open(path) as p:
            self.pages = [re.sub(r"[ \t]+", " ", pg.extract_text() or "") for pg in p.pages]

    def page_of(self, needle):
        n = re.sub(r"\s+", "", needle)
        for i, t in enumerate(self.pages):
            if n in re.sub(r"\s+", "", t):
                return i + 1
        return ""

    @property
    def text(self):
        return "\n".join(self.pages)


rows = []


def add(**kw):
    r = {f: "" for f in FIELDS}
    r.update({k: ("" if v is None else v) for k, v in kw.items()})
    if r["sample_start"] and dt.date.fromisoformat(str(r["sample_start"])) > CUTOFF:
        return
    rows.append(r)


# ---------------------------------------------------------------------------
# 1. FHM outpatient residual sera, final version (updated 2022-12-20)
# ---------------------------------------------------------------------------
def fhm_outpatient(fname, version, adj, study_id):
    pdf = Pdf(RAW / fname)
    t = pdf.text
    wb_url = {"fhm_oppenvarden_wb20230315.pdf": "https://web.archive.org/web/20230315011141id_/" + FHM_OPEN_URL,
              "fhm_oppenvarden_wb20210720.pdf": "https://web.archive.org/web/20210720183637id_/" + FHM_OPEN_URL,
              "fhm_oppenvarden_delrapport1_wb20210118.pdf": "https://web.archive.org/web/20210118042541id_/" + FHM_OPEN_D1_URL}[fname]
    orig = FHM_OPEN_D1_URL if "delrapport" in fname else FHM_OPEN_URL
    src_url = f"{orig} (gone; archived {wb_url})"
    # National table (Tabell 1 in older / Tabell 3 in final): "2020 20-26 april (17) 1397 5,3 (3,82-7,12)"
    tab_nat = re.compile(r"^\s*(20\d\d)\s+([0-9][^()\n]*?)\((\d+(?:-\d+)?)\)\s+(\d[\d ]*\d)\s+(\d+[,.]\d+)\s*\((\d+[,.]\d+)\s*-\s*(\d+[,.]\d+)\)", re.M)
    for m in tab_nat.finditer(t):
        yr, per, wk, n, p, lo, hi = m.groups()
        w = [int(x) for x in wk.split("-")]
        s, e = iso_bounds(int(yr), w[0], w[-1])
        line = m.group(0).strip()
        tab = "Tabell 3" if "wb2023" in fname else "Tabell 1"
        add(study_id=study_id, region_code="SE", region_name="Sweden (9 sampled regions pooled)",
            sample_start=s, sample_end=e, iso_weeks=f"{yr}-W{wk}", period_as_printed=f"{per.strip()} ({wk}) {yr}",
            population_sampled="residual sera from outpatient clinical chemistry, ages 0-95, 9 regions (Jämtland Härjedalen, Jönköping, Kalmar, Skåne, Stockholm, Uppsala, Västerbotten, Västra Götaland, Örebro)",
            n_tested=n.replace(" ", ""), prevalence_pct=num(p), ci_low_pct=num(lo), ci_high_pct=num(hi),
            adjustment=adj, method=FHM_METHOD, report_version=version, source_file=f"raw/sero/{fname}",
            source_url=src_url, table_or_figure=tab, page=pdf.page_of(line.split("(")[0] + "(" + wk + ")"))
    # Region table (Stockholm, VG, Skåne): "2020 20-26 april (17) 8,1 (4,98-12,12) 5,2 (2,28-9,32) 3,6 (1,60-6,44)"
    tab_reg = re.compile(r"^\s*(20\d\d)\s+([0-9][^()\n]*?)\((\d+(?:-\d+)?)\)\s+" + r"\s+".join([r"(\d+[,.]\d+)\s*\((\d+[,.]\d+)\s*-\s*(\d+[,.]\d+)\)"] * 3), re.M)
    reg_tab_start = t.find("Tabell 6" if "wb2023" in fname else "Tabell 4. Procent")
    if reg_tab_start >= 0:
        for m in tab_reg.finditer(t, reg_tab_start):
            yr, per, wk = m.group(1), m.group(2), m.group(3)
            w = [int(x) for x in wk.split("-")]
            s, e = iso_bounds(int(yr), w[0], w[-1])
            vals = m.groups()[3:]
            pg = ""
            for i, ptxt in enumerate(pdf.pages):
                if ("Tabell 6" in ptxt or "Tabell 4. Procent" in ptxt) and vals[0] in ptxt:
                    pg = i + 1
            for k, reg in enumerate(["Stockholm", "Västra Götaland", "Skåne"]):
                p, lo, hi = vals[3 * k:3 * k + 3]
                note = "Västra Götaland ages 0-95 sampled in Göteborg except wk 48-49 2020 when 20-95 sampled in Trollhättan" if reg == "Västra Götaland" else ""
                add(study_id=study_id, region_code=REGION[reg], region_name=reg, sample_start=s, sample_end=e,
                    iso_weeks=f"{yr}-W{wk}", period_as_printed=f"{per.strip()} ({wk}) {yr}",
                    population_sampled="residual sera from outpatient clinical chemistry, ages 0-95",
                    prevalence_pct=num(p), ci_low_pct=num(lo), ci_high_pct=num(hi), adjustment=adj,
                    method=FHM_METHOD, report_version=version, source_file=f"raw/sero/{fname}", source_url=src_url,
                    table_or_figure="Tabell 6" if "wb2023" in fname else "Tabell 4", page=pg,
                    notes=(note + "; n per region not published").strip("; "))
    # Other regions, only in running text for wk 42-43 and 48-49 2020 (and 2021):
    # "Andelen antikroppspositiva i övriga undersökta regioner var 6,1 procent (95 procent KI 3,32-9,90) i Uppsala, ..."
    for sec in re.finditer(r"Insamlingsperioden ([^\n]*?)\(vecka (\d+)-(\d+)\) (20\d\d)(.*?)(?=Insamlingsperioden|\Z)", t, re.S):
        per, w1, w2, yr, body = sec.groups()
        mo = re.search(r"övriga undersökta\s+regioner var(.*?)(?:\.\s|\.$|$)", body, re.S)
        if not mo:
            continue
        seg = re.sub(r"\s+", " ", mo.group(1))
        for m in re.finditer(r"(\d+,\d+) procent \(95 procent KI (\d+,\d+)[- ]\s?(\d+,\d+)\)? i ([A-ZÅÄÖ][a-zåäö]+)", seg):
            p, lo, hi, reg = m.groups()
            s, e = iso_bounds(int(yr), int(w1), int(w2))
            add(study_id=study_id, region_code=REGION[reg], region_name=reg, sample_start=s, sample_end=e,
                iso_weeks=f"{yr}-W{w1}-{w2}", period_as_printed=f"{per.strip()} (vecka {w1}-{w2}) {yr}",
                population_sampled="residual sera from outpatient clinical chemistry, ages 0-95 (Uppsala: ages 0-19 only in 2021)",
                prevalence_pct=num(p), ci_low_pct=num(lo), ci_high_pct=num(hi), adjustment=adj, method=FHM_METHOD,
                report_version=version, source_file=f"raw/sero/{fname}", source_url=src_url,
                table_or_figure="running text (not tabulated)", page=pdf.page_of(f"{p} procent (95 procent KI {lo}"),
                notes="regions with <100 samples in a period are not reported by FHM; n per region not published")
    # Delrapport 1 (Sept 2020) reports regions only in running text, % signs
    if "delrapport" in fname:
        for sec in re.finditer(r"Insamlingsperioden ([^\n]*?)\(vecka (\d+)\)(.*?)(?=Insamlingsperioden|Figur 1|\Z)", t, re.S):
            per, w, body = sec.groups()
            b = re.sub(r"\s+", " ", body)
            s, e = iso_bounds(2020, int(w))
            mn = re.search(r"förekomst av antikroppar på (\d+,\d+)% \(95% KI (\d+,\d+)-(\d+,\d+)\)", b)
            if mn:
                add(study_id=study_id, region_code="SE", region_name="Sweden (9 sampled regions pooled)", sample_start=s,
                    sample_end=e, iso_weeks=f"2020-W{w}", period_as_printed=f"{per.strip()} (vecka {w})",
                    population_sampled="residual sera from outpatient clinical chemistry, ages 0-95, 9 regions",
                    prevalence_pct=num(mn.group(1)), ci_low_pct=num(mn.group(2)), ci_high_pct=num(mn.group(3)),
                    adjustment=adj, method=FHM_METHOD, report_version=version, source_file=f"raw/sero/{fname}",
                    source_url=src_url, table_or_figure="running text", page=pdf.page_of(f"{mn.group(1)}% (95% KI {mn.group(2)}"),
                    notes="preliminary; superseded by later versions of the same report")
            for m in re.finditer(r"(Stockholmsregionen|Stockholm|Västra Götaland|Skåne)[^0-9]{0,40}?(\d+,\d+)%[;,]? \(?(?:95% KI )?(\d+,\d+)-\s?(\d+,\d+)\)?", b):
                reg, p, lo, hi = m.groups()
                if "KI" not in b[m.start():m.end()]:
                    continue
                name = CANON.get(reg, reg)
                add(study_id=study_id, region_code=REGION[reg], region_name=name, sample_start=s, sample_end=e,
                    iso_weeks=f"2020-W{w}", period_as_printed=f"{per.strip()} (vecka {w})",
                    population_sampled="residual sera from outpatient clinical chemistry, ages 0-95",
                    prevalence_pct=num(p), ci_low_pct=num(lo), ci_high_pct=num(hi), adjustment=adj, method=FHM_METHOD,
                    report_version=version, source_file=f"raw/sero/{fname}", source_url=src_url,
                    table_or_figure="running text", page=pdf.page_of(f"{p}%" + b[m.start():m.end()].split(p + "%", 1)[1]),
                    notes="preliminary; superseded by later versions of the same report")


fhm_outpatient("fhm_oppenvarden_wb20230315.pdf", "final: updated 2022-12-20 (archived 2023-03-15)", FHM_ADJ_FINAL, "FHM_outpatient")
fhm_outpatient("fhm_oppenvarden_delrapport1_wb20210118.pdf", "Delrapport 1, updated 2020-09-03", FHM_ADJ_2020, "FHM_outpatient_2020prelim")

# ---------------------------------------------------------------------------
# 2. FHM blood donors
# ---------------------------------------------------------------------------
def fhm_blood(fname, version, study_id):
    pdf = Pdf(RAW / fname)
    t = pdf.text
    ts = {"fhm_blodgivare_wb20221223.pdf": "20221223052503", "fhm_blodgivare_wb20210720.pdf": "20210720183636"}[fname]
    src_url = f"{FHM_BD_URL} (gone; archived https://web.archive.org/web/{ts}id_/{FHM_BD_URL})"
    tab_nat = re.compile(r"^\s*(20\d\d)\s+([0-9][^()\n]*?)\((\d+(?:-\d+)?)\)\s+(\d[\d ]*\d)\s+(\d+[,.]\d+)\s*(?:\((\d+[,.]\d+)\s*-\s*(\d+[,.]\d+)\))?", re.M)
    for m in tab_nat.finditer(t):
        yr, per, wk, n, p, lo, hi = m.groups()
        w = [int(x) for x in wk.split("-")]
        s, e = iso_bounds(int(yr), w[0], w[-1])
        note = ""
        if wk == "12":
            note = "table prints '22-28 mars (12)' although ISO week 12 is 16-22 March; text says week 12-13 = 16-29 March; wk 12-13 sampled only Kalmar, Stockholm, Västerbotten"
        if wk == "13":
            note = "wk 12-13 sampled only Kalmar, Stockholm, Västerbotten"
        add(study_id=study_id, region_code="SE", region_name="Sweden (sampled regions pooled)", sample_start=s, sample_end=e,
            iso_weeks=f"{yr}-W{wk}", period_as_printed=f"{per.strip()} ({wk}) {yr}",
            population_sampled="blood donors (adults), 9 regions from wk 17 (8 regions 2021)", n_tested=n.replace(" ", ""),
            prevalence_pct=num(p), ci_low_pct=num(lo), ci_high_pct=num(hi), adjustment=FHM_BD_ADJ_FINAL, method=FHM_METHOD,
            report_version=version, source_file=f"raw/sero/{fname}", source_url=src_url, table_or_figure="Tabell 1",
            page=pdf.page_of(f"({wk}) {n}"), notes=note)
    # Tabell 2 regional
    i2 = t.find("Tabell 2.")
    if i2 < 0:
        return
    tab2 = t[i2:]
    hdr = re.findall(r"(\d+ \w+-\d+ \w+|\d+-\d+ \w+|\d+ \w+-\s*\d+\s*\w*)", tab2[:600])
    cell = r"(\d+,\d+ \(\d+,\d+-\d+,\d+\)|\*)"
    periods = {"fhm_blodgivare_wb20221223.pdf": [(2020, 48, 49, "23 november-6 december 2020 (48-49)"), (2021, 9, 10, "1-12 mars 2021 (9-10)"),
                                                  (2021, 21, 22, "24 maj-4 juni 2021 (21-22)"), (2021, 38, 39, "20 september-3 oktober 2021 (38-39)")],
               "fhm_blodgivare_wb20210720.pdf": [(2020, 42, 43, "12-25 oktober 2020 (42-43)"), (2020, 48, 49, "23 november-6 december 2020 (48-49)"),
                                                 (2021, 9, 10, "1-12 mars 2021 (9-10)"), (2021, 21, 22, "24 maj-4 juni 2021 (21-22)")]}[fname]
    # verify the column headers we assume are what the PDF prints
    for (_, w1, w2, _) in periods:
        assert f"{w1}-{w2}" in re.sub(r"\s+", "", tab2[:700]).replace("(", "").replace(")", "") or f"({w1}-" in tab2[:700], (fname, w1)
    pg = pdf.page_of("Tabell 2.")
    for line in tab2.splitlines():
        m = re.match(r"\s*(Jönköping|Kalmar|Skåne|Stockholm|Uppsala|Västerbotten|Västra Götaland|Västra|Örebro)\s+(.*)$", line)
        if not m:
            continue
        reg = CANON.get(m.group(1), m.group(1))
        cells = re.findall(cell, m.group(2))
        if len(cells) != len(periods):
            continue
        for (yr, w1, w2, lab), c in zip(periods, cells):
            if c == "*":
                continue
            cm = re.match(r"(\d+,\d+) \((\d+,\d+)-(\d+,\d+)\)", c)
            p, lo, hi = cm.groups()
            note = "regions with <100 samples in a period are not reported"
            if num(lo) > num(p):
                note += f"; PDF prints CI '{lo}-{hi}' (lower bound above estimate, evidently a typo); lower bound left blank"
                lo = None
            s, e = iso_bounds(yr, w1, w2)
            add(study_id=study_id, region_code=REGION[reg], region_name=reg, sample_start=s, sample_end=e,
                iso_weeks=f"{yr}-W{w1}-{w2}", period_as_printed=lab, population_sampled="blood donors (adults)",
                prevalence_pct=num(p), ci_low_pct=num(lo), ci_high_pct=num(hi), adjustment=FHM_BD_ADJ_FINAL,
                method=FHM_METHOD, report_version=version, source_file=f"raw/sero/{fname}", source_url=src_url,
                table_or_figure="Tabell 2", page=pg, notes=note)


fhm_blood("fhm_blodgivare_wb20221223.pdf", "final: updated 2021-10-28 (archived 2022-12-23)", "FHM_blooddonors")
# earlier version kept only for the wk 42-43 2020 regional column, which the final version dropped
_n = len(rows)
fhm_blood("fhm_blodgivare_wb20210720.pdf", "updated 2021-07-19 (archived 2021-07-20)", "FHM_blooddonors_v2021-07")
rows[_n:] = [r for r in rows[_n:] if r["iso_weeks"] == "2020-W42-43" and r["region_code"] != "SE"]
for r in rows[_n:]:
    r["notes"] += "; wk 42-43 regional column appears only in this 2021-07 version (dropped from the final version)"

# Delrapport 2 (Sept 2020): week-24 regional blood donor values in running text
pdf = Pdf(RAW / "fhm_blodgivare_delrapport2_wb20210126.pdf")
b = re.sub(r"\s+", " ", pdf.text)
mm = re.search(r"Vecka 24 var andelen blodgivare.*?figur 2", b)
for m in re.finditer(r"(Stockholm|Väster Götland|Skåne) med (\d+,\d+) procent \(95% KI (\d+,\d+)-(\d+,\d+)\)", mm.group(0)):
    reg, p, lo, hi = m.groups()
    s, e = iso_bounds(2020, 24)
    add(study_id="FHM_blooddonors_2020prelim", region_code=REGION[reg], region_name=CANON.get(reg, reg), sample_start=s,
        sample_end=e, iso_weeks="2020-W24", period_as_printed="vecka 24", population_sampled="blood donors (adults, median age 43-48)",
        prevalence_pct=num(p), ci_low_pct=num(lo), ci_high_pct=num(hi), adjustment=FHM_BD_ADJ_2020, method=FHM_METHOD,
        report_version="Delrapport 2, updated 2020-09-03 (archived 2021-01-26)",
        source_file="raw/sero/fhm_blodgivare_delrapport2_wb20210126.pdf",
        source_url=f"{FHM_BD_D2_URL} (gone; archived https://web.archive.org/web/20210126213855id_/{FHM_BD_D2_URL})",
        table_or_figure="running text (Figur 2 unlabelled)", page=pdf.page_of(f"{p} procent"),
        notes="PDF spells 'Väster Götland'; other weeks/regions only in unlabelled figures, not extracted")

# ---------------------------------------------------------------------------
# 3. FHM news 20 May 2020: first (crude) week-18 results
# ---------------------------------------------------------------------------
nt = open(RAW / "fhm_news_20200520_wb.html", encoding="utf-8", errors="ignore").read()
nt = html.unescape(re.sub(r"<[^>]+>", " ", nt))
nt = re.sub(r"\s+", " ", nt)
m = re.search(r"Analyserna för vecka 18 \(totalt ([\d  ]+) analyserade prover\).*?Stockholm var positiva.*?\.", nt)
seg = nt[m.start(): m.start() + 600]
ntot = re.sub(r"\D", "", m.group(1))
for reg, pat in [("Stockholm", r"Totalt (\d+,\d+) procent av de insamlade blodproverna från personer i Stockholm"),
                 ("Skåne", r"(\d+,\d+) procent i Skåne"), ("Västra Götaland", r"(\d+,\d+) procent i Västra Götaland")]:
    v = re.search(pat, seg).group(1)
    s, e = iso_bounds(2020, 18)
    add(study_id="FHM_news_2020-05-20", region_code=REGION[reg], region_name=reg, sample_start=s, sample_end=e,
        iso_weeks="2020-W18", period_as_printed="vecka 18", population_sampled="residual outpatient sera (same FHM programme)",
        prevalence_pct=num(v), adjustment="crude share of samples positive (as stated in press release)", method=FHM_METHOD,
        report_version="press release 2020-05-20", source_file="raw/sero/fhm_news_20200520_wb.html",
        source_url=f"{NEWS_URL} (archived https://web.archive.org/web/2020id_/{NEWS_URL})", table_or_figure="press release text",
        notes=f"first published figure (the '7.3%' cited widely); {ntot} samples analysed nationally in wk 18; no CI; superseded by FHM_outpatient")

# ---------------------------------------------------------------------------
# 4. Rinkeby-Kista (Stockholm city district), 22-24 June 2020
# ---------------------------------------------------------------------------
pdf = Pdf(RAW / "fhm_rinkeby_kista_2020.pdf")
b = re.sub(r"\s+", " ", pdf.text)
m = re.search(r"Rinkeby-Kista (\d+) (\d+) (\d+,\d+) \((\d+,\d+)-(\d+,\d+)\)", b)
npos, nneg, p, lo, hi = m.groups()
add(study_id="FHM_RinkebyKista", region_code="01", region_name="Stockholm (Rinkeby-Kista district only)", sample_start="2020-06-22",
    sample_end="2020-06-24", iso_weeks="2020-W26", period_as_printed="22-24 juni 2020",
    population_sampled="random population-register sample aged 16-70 in Rinkeby-Kista district, Stockholm city (2153 invited, 538 participated, 530 analysable)",
    n_tested=int(npos) + int(nneg), n_positive=npos, prevalence_pct=num(p), ci_low_pct=num(lo), ci_high_pct=num(hi),
    adjustment="survey-weighted (SCB weights for selection, non-response, calibration) + Rogan-Gladen (sens 100%, spec 99.6%)",
    method="Abbott Architect SARS-CoV-2 IgG (N); weak reactives confirmed with DiaSorin Liaison S1/S2 IgG",
    report_version="FHM report art. 20129 (2020)", source_file="raw/sero/fhm_rinkeby_kista_2020.pdf",
    source_url=f"{RK_URL} (gone; archived https://web.archive.org/web/20200906161950id_/{RK_URL})", table_or_figure="Tabell 9",
    page=pdf.page_of("Tabell 9"), notes="sub-regional (a deprived, high-incidence district); NOT representative of Stockholm län")

# ---------------------------------------------------------------------------
# 5. Castro Dopico et al. 2021, J Intern Med 290:666 (Stockholm blood donors + pregnant women)
# ---------------------------------------------------------------------------
x = open(RAW / "castrodopico2021_fulltext.xml", encoding="utf-8").read()
tab = re.search(r"<table-wrap[^>]*>.*?</table-wrap>", x, re.S).group(0)
t1 = {}
for tr in re.findall(r"<tr>(.*?)</tr>", tab, re.S):
    cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
    if cells and re.match(r"^\d+:", cells[0]):
        wk, dates = cells[0].split(":", 1)
        t1.setdefault(int(wk), []).append((dates.strip(), cells[1:]))
wb = openpyxl.load_workbook(RAW / "castrodopico2021_supp" / "JOIM-290-666-s001.xlsx", read_only=True)
s1 = {int(r[0]): r for r in wb["Bayesian learner"].iter_rows(values_only=True) if isinstance(r[0], int)}
cd_common = dict(region_code="01", region_name="Stockholm", method="in-house IgG ELISA vs SARS-CoV-2 spike trimer and RBD (Karolinska/KI)",
                 source_url=CD_URL)
# Table 1 rows: weeks listed under 2020 first, then 2021 after the '2021' marker
year = 2020
order = []
for tr in re.findall(r"<tr>(.*?)</tr>", tab, re.S):
    txt = html.unescape(re.sub(r"<[^>]+>", " ", tr))
    if re.search(r"^\s*2021", txt):
        year = 2021
    m = re.match(r"\s*(\d+):\s*([^|]+?)\s{2,}", txt)
    cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
    if cells and re.match(r"^\d+:", cells[0]):
        order.append((year, int(cells[0].split(":")[0]), cells[0], cells[1:]))
for yr, wk, label, vals in order:
    s, e = iso_bounds(yr, wk)
    sup = s1.get(wk)
    # Supplementary Table 1 weeks run 14..53 then 1..8; weeks 1-8 belong to 2021
    lo = hi = ""
    note = "Bayesian (cut-off independent) posterior; point estimate from paper Table 1"
    if sup is not None and ((yr == 2020 and wk >= 14) or (yr == 2021 and wk <= 8)):
        lo, hi = round(float(sup[1]) * 100, 2), round(float(sup[5]) * 100, 2)
        note += f"; 95% Bayesian interval from Supplementary Table 1 (fractions x100; S1 median {float(sup[3]) * 100:.2f}%)"
    add(study_id="CastroDopico2021_Bayes", sample_start=s, sample_end=e, iso_weeks=f"{yr}-W{wk:02d}", period_as_printed=f"{label} {yr}",
        population_sampled="random blood donors (n=100/wk) + pregnant women at antenatal screening (n=100/wk), Karolinska Univ. Hospital, Stockholm",
        n_tested=200 if wk != 14 else "", prevalence_pct=num(vals[0]), ci_low_pct=lo, ci_high_pct=hi,
        adjustment="Bayesian probabilistic classifier (no population weighting)", report_version="J Intern Med 2021;290:666-676",
        source_file="raw/sero/castrodopico2021_fulltext.xml; raw/sero/castrodopico2021_supp/JOIM-290-666-s001.xlsx",
        table_or_figure="Table 1 (Bayesian estimate column); Supplementary Table 1", notes=note + "; n=200 per sampled week per Methods (100 donors + 100 pregnant women); week 14 n not stated", **cd_common)
    # S/RBD IgG threshold-based percentages from Table 1
    for j, lab in enumerate(["S IgG (3SD)", "RBD IgG (3SD)", "S IgG (6SD)", "RBD IgG (6SD)"]):
        add(study_id=f"CastroDopico2021_{lab.replace(' ', '').replace('(', '_').replace(')', '')}", sample_start=s, sample_end=e,
            iso_weeks=f"{yr}-W{wk:02d}", period_as_printed=f"{label} {yr}",
            population_sampled="random blood donors + pregnant women, Karolinska Univ. Hospital, Stockholm",
            prevalence_pct=num(vals[j + 1]), adjustment=f"crude % above {lab} cut-off (mean+3SD/6SD of 2019 controls)",
            report_version="J Intern Med 2021;290:666-676", source_file="raw/sero/castrodopico2021_fulltext.xml",
            table_or_figure=f"Table 1 ({lab} column)", notes="no CI given", **cd_common)
# Supplementary Table 2 (SVM-LDA learner) weekly, with CI and N
for r in wb["SVM-LDA learner"].iter_rows(values_only=True):
    if not r or not isinstance(r[2], int) or not str(r[1]).startswith("Wk"):
        continue
    wk = int(str(r[1])[2:])
    yr = 2021 if wk <= 8 else 2020
    s, e = iso_bounds(yr, wk)
    grp = {"Blood donors": "random blood donors", "Pregnant volunteers": "pregnant women at antenatal screening",
           "Combined": "blood donors + pregnant women combined"}[r[0]]
    add(study_id=f"CastroDopico2021_SVMLDA_{r[0].split()[0]}", sample_start=s, sample_end=e, iso_weeks=f"{yr}-W{wk:02d}",
        period_as_printed=f"{r[1]} (Weeks (14) 2020 - (8) 2021)", population_sampled=f"{grp}, Karolinska Univ. Hospital, Stockholm",
        n_tested=r[2], prevalence_pct=round(float(r[3]) * 100, 3), ci_low_pct=round(float(r[4]) * 100, 3), ci_high_pct=round(float(r[5]) * 100, 3),
        adjustment="SVM-LDA probabilistic classifier, equal-weighted (no population weighting)", report_version="J Intern Med 2021;290:666-676",
        source_file="raw/sero/castrodopico2021_supp/JOIM-290-666-s001.xlsx", table_or_figure="Supplementary Table 2",
        notes="fractions in source converted x100; year assigned from 'Weeks (14) 2020 - (8) 2021' header", **cd_common)

# ---------------------------------------------------------------------------
# 6. Roxhed et al. 2021, Nat Commun 12:3695 (home-sampled DBS, random Stockholm residents, April 2020)
# ---------------------------------------------------------------------------
pdf = Pdf(RAW / "roxhed2021_natcommun.pdf")
b = re.sub(r"\s+", " ", pdf.text)
m = re.search(r"from (\d+) random and undiagnosed individuals.*?seroprevalence of (\d+\.\d+)% \(95% CI: (\d+\.\d+)%[-–](\d+\.\d+)%\)", b)
n, p, lo, hi = m.groups()
npos = re.search(r"were (\d+) samples deemed seropositive", b).group(1)
add(study_id="Roxhed2021", region_code="01", region_name="Stockholm (metropolitan)", sample_start="2020-04-01", sample_end="",
    iso_weeks="", period_as_printed="kits mailed during April 2020; 55% returned within 3 weeks",
    population_sampled="random population-register sample aged 20-74, metropolitan Stockholm, self-sampled dried blood spots (2000 invited)",
    n_tested=n, n_positive=npos, prevalence_pct=num(p), ci_low_pct=num(lo), ci_high_pct=num(hi),
    adjustment="classification-based (IgG and/or IgM, multiple antigens); not weighted to population",
    method="multiplex bead IgG/IgM vs S, RBD, N in dried blood spots (KTH/SciLifeLab)", report_version="Nat Commun 2021;12:3695",
    source_file="raw/sero/roxhed2021_natcommun.pdf", source_url=RX_URL, table_or_figure="Abstract / Results",
    page=pdf.page_of("12.5% (95% CI"), notes="sample_start is the first day of April 2020 (paper states only 'during April 2020'); sample_end not stated; includes IgM-only positives (2.1%)")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"wrote {len(rows)} rows to {OUT}")
