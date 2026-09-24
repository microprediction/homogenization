#!/usr/bin/env bash
# Download the seroprevalence sources for Sweden 2020 into raw/sero/, then extract.
# Run from data/sweden/:  bash scripts/fetch_sero.sh
#
# The FHM publications were withdrawn from folkhalsomyndigheten.se (their pages now
# redirect to "Publikationen är borttagen" and the PDF URLs return 404), so each FHM file
# is fetched from the Internet Archive using the raw "id_" form of the snapshot URL.
# Original URL and snapshot timestamp are both recorded below and in sero_estimates.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p raw/sero
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# fetch <outfile> <original_url> <wayback_timestamp or ''>
fetch() {
  local out="raw/sero/$1" orig="$2" ts="$3"
  if [ -s "$out" ]; then echo "have $out"; return; fi
  if curl -sfL -A "$UA" -o "$out" "$orig" && ! grep -qi "borttagen\|<html" "$out"; then
    echo "got $out from original"; return
  fi
  [ -n "$ts" ] || { echo "FAILED $out"; rm -f "$out"; return 1; }
  for i in 1 2 3 4 5; do
    curl -sL --max-time 180 -o "$out" "https://web.archive.org/web/${ts}id_/${orig}" || true
    if ! grep -q "Temporarily Offline\|Gateway Time-out" "$out" 2>/dev/null; then
      echo "got $out from web.archive.org/web/${ts}id_/"; return
    fi
    sleep 10
  done
  echo "FAILED $out"; return 1
}

FHM=https://www.folkhalsomyndigheten.se
# FHM outpatient residual sera ("Påvisning av antikroppar mot SARS-CoV-2 i blodprov från öppenvården")
fetch fhm_oppenvarden_wb20230315.pdf  "$FHM/contentassets/9c5893f84bd049e691562b9eeb0ca280/pavisning-antikroppar-mot-sars-cov-2-blodprov-oppenvarden.pdf" 20230315011141   # final, updated 2022-12-20
fetch fhm_oppenvarden_wb20210720.pdf  "$FHM/contentassets/9c5893f84bd049e691562b9eeb0ca280/pavisning-antikroppar-mot-sars-cov-2-blodprov-oppenvarden.pdf" 20210720183637   # updated 2021-07-19
fetch fhm_oppenvarden_delrapport1_wb20210118.pdf "$FHM/contentassets/9c5893f84bd049e691562b9eeb0ca280/pavisning-antikroppar-genomgangen-covid-19-blodprov-oppenvarden-delrapport-1.pdf" 20210118042541  # Delrapport 1, updated 2020-09-03
# FHM blood donors ("Påvisning av antikroppar mot SARS-CoV-2 hos blodgivare")
fetch fhm_blodgivare_wb20221223.pdf   "$FHM/contentassets/376f9021a4c84da08de18ac597284f0c/pavisning-antikroppar-mot-sars-cov-2-blodgivare.pdf" 20221223052503   # final, updated 2021-10-28
fetch fhm_blodgivare_wb20210720.pdf   "$FHM/contentassets/376f9021a4c84da08de18ac597284f0c/pavisning-antikroppar-mot-sars-cov-2-blodgivare.pdf" 20210720183636   # updated 2021-07-19
fetch fhm_blodgivare_delrapport2_wb20210126.pdf "$FHM/contentassets/376f9021a4c84da08de18ac597284f0c/pavisning-antikroppar-genomgangen-covid-19-blodgivare-delrapport-2.pdf" 20210126213855  # Delrapport 2, updated 2020-09-03
# FHM Rinkeby-Kista random-sample survey, 22-24 June 2020 (art. 20129)
fetch fhm_rinkeby_kista_2020.pdf      "$FHM/contentassets/2cf102cd299c4382b9a0447dc0626356/forekomsten-antikroppar-rinkeby-kista.pdf" 20200906161950
# FHM press release 20 May 2020, first (crude) week-18 results
fetch fhm_news_20200520_wb.html       "$FHM/nyheter-och-press/nyhetsarkiv/2020/maj/forsta-resultaten-fran-pagaende-undersokning-av-antikroppar-for-covid-19-virus/" 2020
# FHM modelling report (not serology; context only): peak day / share infected in Stockholm, Feb-Apr 2020 (art. 20059)
fetch fhm_peakday_model.pdf           "$FHM/contentassets/e1c3b83fa24f4d019e4842053ffd8300/estimates-peak-day-infected-during-covid-19-outbreak-stockholm-feb-apr-2020.pdf" 2020

# Castro Dopico et al., J Intern Med 2021;290:666 (doi:10.1111/joim.13304, PMC8242905): JATS full text + supplement
[ -s raw/sero/castrodopico2021_fulltext.xml ] || curl -sfL -o raw/sero/castrodopico2021_fulltext.xml "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8242905/fullTextXML"
if [ ! -s raw/sero/castrodopico2021_supp/JOIM-290-666-s001.xlsx ]; then
  curl -sfL -o raw/sero/castrodopico2021_supp.zip "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8242905/supplementaryFiles"
  unzip -o -q raw/sero/castrodopico2021_supp.zip JOIM-290-666-s001.xlsx -d raw/sero/castrodopico2021_supp
fi
# Roxhed et al., Nat Commun 2021;12:3695 (doi:10.1038/s41467-021-23893-4)
fetch roxhed2021_natcommun.pdf "https://www.nature.com/articles/s41467-021-23893-4.pdf" ""

for f in raw/sero/*.pdf; do pdftotext -layout "$f" "${f%.pdf}.txt"; done
python3 scripts/extract_sero.py
