# Point-check data: Geneva, England regions, Manaus (first wave, 2020)

Built 2026-09-23. `python fetch_points.py` downloads everything into `raw/` (idempotent). `python build_points.py`
writes the CSVs in this folder. All numbers come from downloaded files: CSV, XLSX, JATS XML full text, or text that
`pdftotext -layout` extracts from supplementary PDFs. Nothing was typed in by hand. The one exception to scripted
download is noted under Manaus. The whole `data/` tree is gitignored, so these scripts are not under version
control unless they are copied out.

## Outputs

| file | contents |
|---|---|
| `geneva_daily.csv` | 2020 daily: FOPH deaths by date of death and hospital admissions by admission date (`deaths`, `hosp_admissions`); cantonal series from openZH; deaths from the Perez-Saez et al. IFR study (all, and care-home residents) |
| `geneva_serology.csv` | SEROCoV-POP weeks 1-5: main adjusted estimates, raw positive shares, and appendix sensitivity analyses |
| `england_daily.csv` | 2020 daily: deaths by date of death for the 9 ONS regions and England; admissions for the 7 NHS England regions and England |
| `england_serology.csv` | REACT-2 round 1 by ONS region (crude, test-adjusted, test-adjusted and weighted); ONS CIS antibody by region (two releases) |
| `manaus_daily.csv` | 2020 daily for Manaus and Amazonas: reported COVID deaths (wcota), SIM all-cause deaths and B34.2 deaths by date of death, same-calendar-day all-cause counts for 2015-2019 |
| `manaus_sim_allcause_by_year.csv` | SIM all-cause and B34.2 deaths by date of death, 2015-2020, for Manaus and Amazonas (residence) |
| `manaus_serology.csv` | Buss et al. Table S2, monthly blood-donor prevalence for Manaus (and São Paulo), every estimate type |
| `populations.csv` | Populations for every place, with vintage and source |
| `owid_subset.csv` | Switzerland, United Kingdom, Brazil, Sweden, Spain: OWID deaths (WHO weekly), OWID JHU daily deaths, Oxford stringency index, population |

Serology column conventions: `prevalence_pct`, `ci_low_pct`, `ci_high_pct` are in percent. `adjusted` is `raw` for
crude shares and `adjusted` otherwise. `method` names the specific adjustment. `n_tested` and `n_positive` are
copied from the source table.

## 1. Geneva

**Serology.** Stringhini S, Wisniak A, Piumatti G, et al. Seroprevalence of anti-SARS-CoV-2 IgG antibodies in
Geneva, Switzerland (SEROCoV-POP): a population-based study. Lancet 2020;396:313-19.
doi:10.1016/S0140-6736(20)31304-0, PMCID PMC7289564.
- Full text XML: `https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:7289564&metadataPrefix=pmc`
  (`raw/PMC7289564.xml`). **Table 2** ("Overview of seroprevalence estimates by week") gives the main estimates. Each
  week has an adjusted estimate with a 95% credible interval, and a raw row holding the positive share from the same
  table (no CI in the source).
- Appendix `mmc1.pdf` from Europe PMC `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7289564/supplementaryFiles`
  (`raw/supp/PMC7289564/mmc1.pdf`). **Table S2** (Bus Santé participants only, appendix p. S2), **Table S4** (ELISA
  cutoff 1.5), **Table S5** (rIFA for everyone with OD ratio ≥0.5), appendix p. S3.
- Caveat on dates: the paper gives only the overall window, 6 April to 9 May 2020, for weeks 1-5. It does not give
  dates for each week. `date_start`/`date_end` are therefore consecutive 7-day blocks from Monday 6 April, with the
  last block capped at 9 May, and `date_basis` says so. `ref_date` is the date the authors assigned to each week in
  their own IFR code (`serodata` in `scripts/infer_ifr.R` of github.com/HopkinsIDD/sarscov2-ifr-gva, saved as
  `raw/hopkins_ifr_gva/infer_ifr.R`): 06 Apr, 16 Apr, 21 Apr, 29 Apr, 06 May. The prevalence values in that R file
  come from an earlier model run and differ from Table 2, so they are not used.
- The sample is aged 5 and over and excludes care-home residents in practice. Perez-Saez et al. note that half of
  the deaths among people aged 65 and over were residents of assisted care facilities.
- The survey continued after week 5, but this paper and its appendix report only weeks 1-5. No later weekly
  estimates were collected here.

**Deaths and hospitalisations.**
- `deaths`, `hosp_admissions`: Swiss FOPH, final open-data release 20231206-0sxi4s4a.
  `https://covid19.admin.ch/api/data/20231206-0sxi4s4a/sources/COVID19Death_geoRegion.csv` and
  `.../COVID19Hosp_geoRegion.csv`, rows `geoRegion == GE`, column `entries`. Deaths are laboratory-confirmed and
  indexed by date of death. Hospitalisations are laboratory-confirmed cases hospitalised, indexed by admission date,
  from mandatory clinical reports (version 2023-01-24). Total for 1 Mar to 30 Jun 2020: 288 deaths and 381
  admissions.
- `deaths_cumulative_canton`, `deaths_canton_daily_diff`, `hosp_new_canton`: openZH/covid_19,
  `https://raw.githubusercontent.com/openZH/covid_19/master/COVID19_Fallzahlen_CH_total_v2.csv`, canton GE, columns
  `ncumul_deceased` and `new_hosp`. The listed source is https://infocovid.smc.unige.ch/. Indexing is by report date.
  Caveats: this series is higher than FOPH (375 deaths from 1 Mar to 30 Jun, and 1,201 new hospitalisations
  against 381). `new_hosp` stops on 2020-06-15. The same rows carry an `ncumul_conf` of about 23,000 by May. That
  cannot be the count of confirmed cases (the paper cites 5,160 on 9 May), which suggests the Geneva series in openZH
  was back-filled under different definitions. Use it only as a cross-check.
- `deaths_ifr_study`, `deaths_ifr_study_care_homes`: Perez-Saez et al., Lancet Infect Dis 2021;21:e69-70,
  doi:10.1016/S1473-3099(20)30584-3. Data file
  `https://raw.githubusercontent.com/HopkinsIDD/sarscov2-ifr-gva/master/data/stratified_dgs_data.csv`, `var ==
  death_incid`, summed over age classes, with `EMS == 1` for care-home residents. It covers 26 Feb to 1 Jun 2020:
  286 deaths, 134 of them in care homes. The paper does not state in this file whether dates are dates of death or
  of report.

**Population.** 509,448 from the FOPH `pop` column (reference year not stated in the file). 506,765 from the IFR
study (`stratified_pop_data.csv`, the sum of its age classes, matching the text "population of 506 765").

## 2. England

**REACT-2 round 1.** Ward H, Atchison C, Whitaker M, et al. SARS-CoV-2 antibody prevalence in England following
the first peak of the pandemic. Nat Commun 2021;12:905. doi:10.1038/s41467-021-21237-w, PMCID PMC7876103.
- Supplementary Information `41467_2021_21237_MOESM1_ESM.pdf` from
  `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7876103/supplementaryFiles`. **Supplementary Table 1**
  (Supplementary information pp. 1-2), England row and Region block. Each region has three estimates with 95% CIs:
  crude prevalence (`raw`), adjusted for test sensitivity and specificity, and adjusted plus re-weighted for sample
  design and response.
- Fieldwork ran 20 June to 13 July 2020. The test was a self-administered IgG lateral flow immunoassay. The sample
  was adults aged 18+ living in the community, so care homes are excluded. n = 99,908 valid results.
- Regions are the nine ONS regions (Government Office Regions), not NHS England regions. The supplement labels one
  "Yorkshire", renamed here to "Yorkshire and The Humber".
- The same supplement (p. 11) prints mid-2019 regional populations, which match the original ONS release in
  `populations.csv`.

**ONS COVID-19 Infection Survey.** Dataset "Coronavirus (COVID-19) Infection Survey: England", 2020 editions:
- Release of 28 Aug 2020: `https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/healthandsocialcare/conditionsanddiseases/datasets/coronaviruscovid19infectionsurveydata/2020/previous/v20/covid19infectionsurveydatasets20200828.xlsx`.
  Table 3a (England, weighted) and **Table 3b**, "Percentage of individuals ever testing positive for antibodies
  (unweighted) by region between 26 April and 9 August 2020".
- Release of 9 Oct 2020: same path with `v27/covid19infectionsurveydatasets20201009.xlsx`. Table 3a and **Table 3b**,
  "...(weighted) by region, England between 26 April and 8 September 2020".
- Proportions in the source are converted to percent. Coverage is ages 16+ in private households. The sheet notes
  say the regional estimates come from Bayesian modelling with credible intervals. The regional samples are small
  (203-1,560 people per region). Note 5 of the October sheet says the weighted regional estimates cannot be directly compared with the earlier unweighted ones.
- The regional weekly modelled antibody series in later ONS releases starts on 7 Dec 2020, after the second wave
  began, so it was not used.

**Deaths and admissions.** The legacy "Coronavirus (COVID-19) in the UK" dashboard has been decommissioned. Its
archive is at https://ukhsa-dashboard.data.gov.uk/covid-19-archive-data-download:
`https://archive.ukhsa-dashboard.data.gov.uk/coronavirus-dashboard/deaths.zip` and `.../healthcare.zip` (files
dated Oct-Dec 2024).
- `deaths` = `Deaths/region_newDailyNsoDeathsByDeathDate.csv` (and the `nation_` file for England). These are ONS
  death registrations with COVID-19 mentioned on the certificate, by date of death, for ONS regions. They include
  deaths in care homes. England total for Feb to Jul 2020 is 49,660.
- `deaths_28d_positive_test` = `Deaths/region_newDeaths28DaysByDeathDate.csv`. These are deaths within 28 days of a
  positive test, by date of death (England total 36,721 for Feb to Jul 2020). Testing was limited early in the
  wave, so this series undercounts then.
- `hosp_admissions` = `Healthcare/nhsRegion_newAdmissions.csv` (and `nation_newAdmissions.csv`), for NHS England
  regions from 2020-03-17. In the NHS England definition, this counts patients admitted with COVID-19 plus inpatients
  newly diagnosed in the past 24 hours. It therefore includes some hospital-acquired infections.
- Geography gap: the archive has deaths only by ONS region and admissions only by NHS region. NHS "Midlands" roughly
  equals East plus West Midlands, and "North East and Yorkshire" roughly equals North East plus Yorkshire and The
  Humber, but the boundaries follow CCGs, not local authorities, so the two do not align exactly. No aggregation was
  done here.
- The current UKHSA dashboard API (api.ukhsa-dashboard.data.gov.uk) has `COVID-19_deaths_ONSByDay` by region. Its
  NHS-region admissions start only on 2020-08-01, which is why the archive is used.

**Populations** (`populations.csv`):
- ONS regions, mid-2019, original June 2020 release, `ukmidyearestimates20192020ladcodes.xls` sheet "MYE2 - Persons"
  (from `https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland/mid2019april2020localauthoritydistrictcodes/ukmidyearestimates20192020ladcodes.xls`).
  Reading this file needs `xlrd`. Without it, the build skips these rows.
- ONS regions, mid-2019 and mid-2020, current vintage rebased on Census 2021, via Nomis NM_2002_1
  (`https://www.nomisweb.co.uk/api/v01/dataset/NM_2002_1.data.csv?geography=TYPE480&date=2019,2020&gender=0&c_age=200&measures=20100`).
  These differ from the original release. London mid-2019, for example, is 8,889,743 here against 8,961,989
  originally.
- NHS England regions, mid-2019: ONS SAPE22DT6a (mid-2019 estimates for 2020 CCGs), "Mid-2019 Persons", summed
  by `NHSER20 Name`
  (`.../clinicalcommissioninggroupmidyearpopulationestimates/mid2019sape22dt6a/sape22dt6amid2019ccg2020estimatesunformatted.zip`).

## 3. Manaus

**Serology.** Buss LF, Prete CA Jr, Abrahim CMM, et al. Three-quarters attack rate of SARS-CoV-2 in the Brazilian
Amazon during a largely unmitigated epidemic. Science 2021;371:288-292. doi:10.1126/science.abe9728, PMCID
PMC7857406.
- **Table S2** of the Supplementary Materials, `abe9728_buss_sm.pdf` (supplement p. 17), from
  `https://www.science.org/doi/suppl/10.1126/science.abe9728/suppl_file/abe9728_buss_sm.pdf`. science.org refuses
  scripted requests (HTTP 403 bot protection), and Europe PMC's supplementary bundle for this article holds only the
  figures. The PDF was therefore fetched once in a normal browser session and saved to
  `raw/buss_science/abe9728_buss_sm.pdf` (1,899,451 bytes, SHA-256
  e50bdcd3136a5aa98166cc6e477943da4405ae3bda765cff5fb3237e126ccc43). The table is parsed from that file by script.
- Each month has these rows. At S/C threshold 1.4: crude (`raw`), age-sex weighted, weighted plus
  sensitivity/specificity adjusted (Rogan-Gladen, sensitivity 84.0%, specificity 99.9%), and, from March on,
  weighted plus adjusted plus seroreversion-corrected (the cumulative attack rate; `adjusted (seroreversion)`). At
  threshold 0.4: crude, weighted, and adjusted (sensitivity 92.2%, specificity 96.7%). Sampling windows run from
  "Feb 7th-13th" to "Oct 10th-17th" 2020.
- The source has typographic slips that were copied as printed: São Paulo June "11. 9" is read as 11.9, and São
  Paulo August weighted-adjusted is "14.1 (11.4 -14.3)", a CI that does not bracket its estimate.
- The sample is blood donors, who are not a random sample of the population; the authors extrapolate cautiously to ages 16-69. The authors' Dryad dataset
  (doi:10.5061/dryad.c59zw3r5n; it contains `plot_df_prevalences.csv`, `stand_deaths.csv`,
  `correction Manaus.csv` and others) sits behind bot protection (Anubis "BotStopper" challenge) and was not
  downloaded.

**Deaths.**
- `deaths_covid_reported` and `..._cumulative`: wcota/covid19br,
  `https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time_2020.csv.gz` (Manaus, ibgeID
  1302603) and `cases-brazil-states.csv` (state AM). Columns are `newDeaths` and `deaths`. These are confirmed COVID
  deaths by report date, compiled from state health secretariat and Ministry of Health bulletins.
- `deaths_all_cause`, `deaths_underlying_B342`: the Ministry of Health mortality information system (SIM), national
  death-certificate microdata `https://diaad.s3.sa-east-1.amazonaws.com/sim/Mortalidade_Geral_{2015..2020}.csv`
  (the DATASUS bucket behind the OpenDataSUS "SIM" dataset). `fetch_points.py` streams each national file and keeps
  only rows with residence or occurrence in Amazonas (`raw/sim/sim_AM_YYYY.csv`, 8 columns). Counts are by date of
  death (`DTOBITO`). "Manaus" means residence `CODMUNRES == 130260`, and "Amazonas" means residence code starting
  13. `deaths_underlying_B342` counts rows whose underlying cause `CAUSABAS == B342`, the code SIM used for COVID-19
  in 2020. It is not the full COVID count, because U07.1/U07.2 markers in the cause lines were not parsed.
- `deaths_all_cause_{2015..2019}_same_day`: raw all-cause counts from those years on the same month-day, supplied so
  a baseline can be formed. No baseline or excess-death figure is computed here. The 2017 series includes the
  1 January 2017 Compaj prison massacre (89 deaths that day).
- Monthly sums for Manaus residents in April 2020: 2,947 all-cause deaths (2015-2019 same-month counts 875-956),
  1,145 with underlying cause B34.2, and 310 COVID deaths reported by report date. This is the undercounting noted
  in the brief.
- Burial and cemetery records (Manaus SEMULSP) and the civil-registry portal (Portal da Transparência do Registro
  Civil) were not collected. SIM all-cause deaths cover the same ground from the primary registry.

**Hospital admissions**: not obtained. SIVEP-Gripe (SRAG) 2020 microdata would provide them, but the OpenDataSUS
download (`opendatasus.saude.gov.br`) did not respond to scripted access, and the S3 paths tried returned 403. The
`hosp_admissions` column in `manaus_daily.csv` is empty.

**Population.** IBGE SIDRA API, table 6579 (estimated resident population, 1 July), variable 9324:
`https://servicodados.ibge.gov.br/api/v3/agregados/6579/periodos/2019%7C2020/variaveis/9324?localidades=N6%5B1302603%5D%7CN3%5B13%5D`.
Manaus: 2,182,763 (2019) and 2,219,580 (2020). Amazonas: 4,144,597 (2019) and 4,207,714 (2020).

## 4. OWID national subset (`owid_subset.csv`)

- `https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv`, filtered to the
  five countries. Kept columns: `iso_code, location, date, population, new_deaths, new_deaths_smoothed,
  total_deaths, new_deaths_smoothed_per_million, stringency_index`.
- Caveat: the current OWID file takes deaths from WHO as weekly totals. `new_deaths` is zero on six days of each
  week and the whole week's total on Sundays, even in 2020. It is renamed `new_deaths_who_weekly` (with
  `new_deaths_smoothed_who_weekly`), and `total_deaths` is also weekly-stepped.
- A daily series is therefore added from OWID's archived JHU CSSE compilation,
  `https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/full_data.csv`: `new_deaths_jhu_daily`
  and `weekly_deaths_jhu_trailing7d`. These are deaths by report date and run to 2023-03-09.
- `stringency_index` is the Oxford COVID-19 Government Response Tracker (OxCGRT) index as carried by OWID. It is
  national, not subnational, and runs to 2022-12-31.

## Gaps, summarised
- Geneva: no per-week dates in the SEROCoV-POP paper (the dates used are inferred and flagged). Only weeks 1-5 are
  covered.
- England: deaths and admissions use different regional geographies (ONS regions against NHS regions). ONS CIS
  regional antibody estimates start 26 April, pool the period to August or September, and have small samples.
- Manaus: no hospital admissions. No burial or civil-registry series. The Dryad data for Buss et al. was not
  retrieved (bot protection). The seroprevalence file covers only what Table S2 prints.
