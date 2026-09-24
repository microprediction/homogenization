# Spain, COVID-19 in 2020 (spring wave and autumn wave), province level: sources

Everything here is rebuilt by `python3 build_spain.py`. It downloads the primary files (it skips files already present; `--refresh` downloads them again) and writes the CSVs. No number was typed by hand. The only hand-written tables in the script are code mappings: INE province code to ISO 3166-2:ES code, plus two ENE-COVID name spellings.

**Reading tip:** the ISO codes `NA` (Navarra) and `NC` (no consta) become NaN in pandas' default CSV reader. Use `pd.read_csv(..., keep_default_na=False)`, or join on `ine_code`.

## Output files

| file | content |
|---|---|
| `spain_provinces.csv` | One row per province (50 provinces plus Ceuta and Melilla, 52 rows). Columns: ISO code, INE code, INE name, population on 1 Jan 2020, ENE-COVID round-3 seroprevalence (%) with 95% CI, round and dates, method, number of participants, source table and page, note. Also CNE deaths and hospital admissions summed over 2020-02-01..2020-07-31 (the first wave). **Round-4 columns (autumn wave):** `sero_r4_*` (current prevalence in round 4), `seroconv_r4_*` (seroconversion among people seronegative in rounds 1-3) and `sero_global_r1_r4_*` (ever positive in rounds 1-4, round-4 participants). Each has pct, CI low, CI high and n; see section 1b. |
| `spain_sero_round4.csv` | ENE-COVID round-4 province tables in long format: 52 areas x 4 measures. Only the "Total" column group is kept; the tables also split by sex. Columns include source table and page. |
| `spain_sero_all_rounds.csv` | ENE-COVID Tabla 4 in long format: 52 areas x rounds 1, 2 and 3. Columns: n, %, CI and round dates. |
| `spain_daily.csv` | One row per date and province of residence, **2020-02-01..2020-12-31** (53 codes including `NC`). Columns: deaths, hospital admissions, ICU admissions and cases, summed over sex and age. |
| `spain_weekly_allcause.csv` | INE EDeS weekly all-cause deaths by province, 2015-2020. Also includes a **derived** same-week 2015-2019 mean and 2020 minus that mean (see caveats). |

## 1. Seroprevalence: ENE-COVID

- **File:** `ESTUDIO_ENE-COVID19_INFORME_FINAL.pdf` ("Estudio ENE-COVID: Informe final", Ministerio de Sanidad / ISCIII, 6 July 2020; the PDF was created on 15 July 2020, 32 pages).
  URL: https://www.sanidad.gob.es/areas/alertasEmergenciasSanitarias/alertasActuales/nCov/ene-covid/docs/ESTUDIO_ENE-COVID19_INFORME_FINAL.pdf
  (The ENE-COVID index page is https://www.sanidad.gob.es/areas/alertasEmergenciasSanitarias/alertasActuales/nCov/ene-covid/home.htm. La Moncloa hosts another copy at https://www.lamoncloa.gob.es/serviciosdeprensa/notasprensa/sanidad14/documents/2020/060720-estudio_ene-covid_informe_final.pdf.)
- **Table used:** "Tabla 4: Prevalencia de anticuerpos IgG anti SARS-CoV-2 por provincia (test rápido)", on **PDF page 15**. The table has 52 rows, and each row gives rounds 1, 2 and 3 as N, %, 95% CI. The script extracts it with `pdftotext -layout -f 15 -l 15`.
- **Rounds** (report p.4): R1 27 Apr-11 May 2020; R2 18 May-1 Jun 2020; R3 8-22 Jun 2020. `spain_provinces.csv` uses **round 3**, the last round of the first wave.
- **Method:** IgG band of the point-of-care lateral-flow test ("test rápido", Orient Gene Biotech COVID-19 IgG/IgM; report p.5, methods). The estimates are weighted for sampling design and non-response (see the protocol).
- **Immunoassay by province: NOT obtained.**
  - The final report gives provincial figures **only for the rapid test**.
  - On p.12 the report says round-3 immunoassay (Abbott ARCHITECT CMIA) results were incomplete. It shows provincial immunoassay values only as maps (Figura 4, rounds 1-2), with no table.
  - The Lancet paper's appendix (Pollán et al. 2020, doi:10.1016/S0140-6736(20)31483-5, PMC7336131, `mmc1.pdf`) should contain the round-1 provincial immunoassay estimates. Automated downloads from PMC, Europe PMC and thelancet.com were all refused (bot challenge / 403), so it is not included. Fetching it needs a browser: https://pmc.ncbi.nlm.nih.gov/articles/PMC7336131/ , supplementary "mmc1.pdf".
- **Supporting copies kept:**
  - `ESTUDIO_ENE-COVID19_SEGUNDA_RONDA_INFORME_PRELIMINAR.pdf` (round-2 preliminary report, Tabla 5 by province, rounds 1-2, rapid test; its round-1 values differ slightly from the final report's).
  - `ene_covid19_final_17.pdf` (ISCIII regional final report for La Rioja, 17 July 2020, https://portalcne.isciii.es/enecovid19/informes/ene_covid19_final_17.pdf). It gives region-level immunoassay figures for R1 and R2 only.
- **Printing issues in Tabla 4, kept as printed:**
  - Ceuta, round 2: Tabla 4 prints CI "0,1 - 8,8". The CCAA table (Tabla 3, p.11) prints "0,1 - 1,7" for the same area and round. This is flagged in `sero_note` / `note`. It does not affect round 3.
  - Melilla, rounds 1 and 2: the dash between the CI bounds is missing. Parsed correctly.
  - La Rioja, round 1: N is 1324 in Tabla 4 and 1323 in Tabla 3.
  - The other single-province regions (Madrid, Asturias, Baleares, Cantabria, Murcia, Navarra) agree between Tabla 3 and Tabla 4.

## 1b. Seroprevalence in the autumn wave: ENE-COVID round 4

- **File:** `ENE-COVID_CUARTA_RONDA.pdf` ("Estudio ENE-COVID: Cuarta ronda", Ministerio de Sanidad / ISCIII, dated 15 December 2020, 37 pages).
  URL: https://www.sanidad.gob.es/gabinetePrensa/notaPrensa/pdf/15.12151220163348113.pdf
- **Fieldwork:** 16-29 November 2020 (report p.1). There were 51,409 participants. The report calls itself **preliminary**: it uses the rapid test only, because laboratory immunoassay results were not yet available (p.1).
- **Method:** same as rounds 1-3, the IgG band of the Orient Gene point-of-care rapid test (p.5). Round 4 is reported **by province**, so no fallback to region tables was needed.
- **Tables used.** The script parses the "Total" column group (N, %, 95% CI) of each table. The tables also give men and women separately.

| `measure` in `spain_sero_round4.csv` | table | PDF page | meaning |
|---|---|---|---|
| `current_r4` | Tabla 4, "Prevalencia actual de anticuerpos IgG anti SARS-CoV-2 por provincia, Ronda 4" | 13 | IgG positive at round 4 |
| `global_r1_to_r4_r4participants` | Tabla 6, "Prevalencia global (Rondas 1-4) ... por provincia" | 16 | positive in any of rounds 1-4, among round-4 participants |
| `seroconversion_r4_among_seronegative_r1_r3` | Tabla 12, "Incidencia de seroconversión (nuevos IgG+) en personas seronegativas en las tres primeras Rondas de ENE-COVID por provincias" | 26 | IgG+ at round 4, among people seronegative in the earlier rounds |
| `global_r1_to_r4_whole_cohort` | Tabla 4b (annex), "Prevalencia global (Rondas 1-4) ... por provincia en el total de la cohorte" | 36 | ever positive, whole cohort of all rounds |

  The regional (CCAA) versions are Tabla 3 (p.12) and Tabla 5 (p.16). They were not extracted.
- **Caveats, from the report:**
  - **Current prevalence falls with antibody waning.** In some high first-wave provinces it is *lower* in round 4 than in round 3: Soria 14.4 to 13.4, Cuenca 11.4 to 11.1, Ciudad Real 9.1 to 8.3. So it is not a cumulative attack rate.
  - **Seroconversion (Tabla 12) is incidence among people seronegative in rounds 1-3.** It measures infection between June and mid-November 2020 among those susceptible after the first wave. The national figure is 3.8% (3.5-4.1) (p.24).
  - **Round 4 was taken during the autumn wave, not after it.** Hospitalisations and deaths were still high in late November and December (see `spain_daily.csv`). IgG takes about 2 weeks to appear, so round 4 reflects infections up to roughly early November.
  - **The two global-prevalence tables bound the true figure** (report note, annex p.33).
    - Tabla 6 may **overestimate** it, because people with an earlier positive result took part in round 4 more often (77.6% vs 75.1% adherence).
    - Tabla 4b may **underestimate** it, because it cannot count seroconversions among the more than 19,000 earlier seronegatives who skipped round 4.
- **Printing issue, kept as printed:** in Melilla's rows the dash between the CI bounds is missing. Parsed correctly.

## 2. Daily deaths, hospital admissions, ICU admissions, cases: ISCIII / CNE (RENAVE, SiViES)

- **File:** `casos_hosp_uci_def_sexo_edad_provres.csv`, from https://cnecovid.isciii.es/covid19/resources/casos_hosp_uci_def_sexo_edad_provres.csv. The file covers 2020-01-01..2022-03-27 and was downloaded 2026-09-23. `spain_daily.csv` keeps 2020-02-01..2020-12-31.
- **Metadata:** `metadata_casos_hosp_uci_def_sexo_edad_provres.pdf` (https://cnecovid.isciii.es/covid19/resources/metadata_casos_hosp_uci_def_sexo_edad_provres.pdf).
- **Caveats, from the metadata:**
  - **Province is the province of residence** (`provincia_iso`, ISO 3166-2:ES). `NC` means not recorded. `NC` carries 186 deaths and 1,426 admissions over Feb-Jul 2020 and is kept as its own row.
  - **Dates are event dates, not report dates.**
    - Deaths are dated by **date of death**. When that is missing, the date of diagnosis is used, and failing that the "fecha clave".
    - Hospitalisations are dated by **date of admission**, with the same fallbacks.
    - ICU admissions are dated by date of ICU admission.
    - Cases before 11 May 2020 are dated by date of diagnosis, then date of notification to the region, then fecha clave.
    - Until 10 May, fecha clave was recommended to be the symptom-onset date.
  - Before 11 May 2020, cases include people who were hospitalised, admitted to ICU or died with a **clinical** COVID-19 diagnosis and no positive test. After 11 May, cases need a positive active-infection test.
  - **Deaths are confirmed or notified COVID-19 cases only** and undercount first-wave mortality, above all in care homes during March-April 2020. Over Feb-Jul 2020 they total 29,950 nationally. The derived INE excess for ISO weeks 10-26 totals about 49,300. The CNE/excess ratio varies by province, roughly 0.5 in Madrid, Barcelona, Albacete and León versus about 0.9 in Bizkaia and Zaragoza. These figures were computed from the files here and are a diagnostic, not a published statistic.
  - `num_hosp` counts COVID-19 cases that were hospitalised, dated at admission. It is not a hospital-census series.
  - The file is summed over sex and age. After 28 March the age groups are only 60+ (per the metadata footnote); totals are unaffected.
  - Daily totals, Feb-Jul 2020: deaths 29,950; hospital admissions 113,540; ICU 9,712; cases 304,810. Aug-Dec 2020: 24,765 deaths and 124,479 hospital admissions.
  - From 11 May 2020 testing was widespread and cases need a positive active-infection test, so **cases are far more complete in the autumn wave than in the spring**. Case counts are not comparable across the two waves.
- Not used: datadista/datasets on GitHub. Its first-wave deaths are by autonomous community and date of report, not by province, so the CNE file is the better province-level source.

## 3. Population

- **File:** `ine_2852.csv`, INE table 2852, "Población por provincias y sexo" (Cifras oficiales de población de los municipios españoles: revisión del Padrón Municipal).
  URL: https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/2852.csv (table page: https://www.ine.es/jaxiT3/Tabla.htm?t=2852).
- Column `Periodo = 2020` is the padrón at **1 January 2020**. Both sexes are used. National total: 47,450,795.

## 4. Excess / all-cause deaths

- **File:** `ine_edes_35176.csv`, INE EDeS table 35176, "Defunciones semanales, acumuladas y variación interanual del acumulado. Total nacional y provincias".
  URL: https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/35176.csv (table page: https://www.ine.es/jaxiT3/Tabla.htm?t=35176; methodology: https://www.ine.es/dynt3/metadatos/es/RespuestaPrint.html?oper=406).
- Only `Tipo de dato = Dato base` (weekly count) is used.
  - Weeks are calendar weeks, Monday-Sunday; INE period code `YYYYSMww`.
  - Deaths are by **date of death** and **province of residence**. INE switched its provisional estimates to residence in May 2024. Years that are definitive, such as 2015-2020, come from the MNP definitive death statistics.
  - Deaths of non-residents are in a separate "No residente" line, which is excluded.
- `mean_2015_2019` and `excess_2020_vs_mean_2015_2019` are **derived by this script**: the plain same-week mean of 2015-2019 and the 2020 difference from it. They are not an official INE or MoMo excess estimate, and there is no trend or age adjustment.
- MoMo (ISCIII) excess estimates exist only for Spain nationally and by autonomous community, **not by province**, so they were not used.

## 5. Interventions (dates only)

- **State of alarm:** Real Decreto 463/2020, de 14 de marzo (BOE-A-2020-3692), published and in force 14 March 2020. https://www.boe.es/buscar/act.php?id=BOE-A-2020-3692
- **Stricter lockdown:** Real Decreto-ley 10/2020, de 29 de marzo (BOE-A-2020-4166). It introduced a recoverable paid leave for all non-essential employees "entre el 30 de marzo y el 9 de abril de 2020, ambos inclusive". https://www.boe.es/buscar/act.php?id=BOE-A-2020-4166

## 5b. Autumn 2020 interventions (dates only)

- **Madrid, 9-24 October 2020:** Real Decreto 900/2020, de 9 de octubre (BOE-A-2020-12109). A state of alarm restricting entry and exit for nine Madrid-region municipalities: Alcobendas, Alcorcón, Fuenlabrada, Getafe, Leganés, Madrid, Móstoles, Parla and Torrejón de Ardoz. It lasted 15 days and ended 24 October 2020, per the BOE consolidated-text note. https://www.boe.es/buscar/act.php?id=BOE-A-2020-12109
- **National state of alarm, from 25 October 2020:** Real Decreto 926/2020, de 25 de octubre (BOE-A-2020-12898), in force on publication, 25 October 2020. https://www.boe.es/buscar/act.php?id=BOE-A-2020-12898
  - Article 5 sets a **night curfew, 23:00-06:00**.
  - Regional authorities could also restrict entry and exit of their territory and cap group sizes.
  - The preamble notes the Canary Islands were at medium risk, so night-mobility restrictions were not considered necessary there at that time.
  - This was **not** a home-confinement lockdown like the spring one.
- **Extension to 9 May 2021:** Real Decreto 956/2020, de 3 de noviembre (BOE-A-2020-13494), extended it from 00:00 on 9 November 2020 to 00:00 on 9 May 2021. https://www.boe.es/buscar/act.php?id=BOE-A-2020-13494
- **Regional measures were not compiled.** These include perimeter closures of regions and municipalities, hospitality closures and curfew-hour changes; they were numerous, differed by region and changed week to week. Each region's official bulletin (BOCM, DOGC, BOJA, etc.) is the primary source.

## Gaps

- No provincial immunoassay seroprevalence for any round (see sections 1 and 1b). Round 4 was published with rapid-test results only. The provincial rapid-test estimates are complete for all 52 areas and all 3 rounds.
- No official provincial excess-mortality estimate. Only raw INE weekly deaths are provided, plus the simple derived baseline.
- CNE deaths undercount first-wave mortality in a way that differs by province (see section 2).
