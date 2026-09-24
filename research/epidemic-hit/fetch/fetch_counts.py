#!/usr/bin/env python3
"""Download primary/archived files for Swedish regional COVID-19 counts (2020).

Everything lands in ../raw/. Re-runnable; existing files are skipped unless --force.
Sources (see ../SOURCES.md for details):
  1. FHM Folkhalsomyndigheten_Covid19.xlsx, final version (23 Mar 2023) via Wayback
     (the ArcGIS item b5e7488e117749c19881cce45db13f7e now returns 403 'Subscription is canceled').
  2. FHM Folkhälsodata PxWeb API tables (live, FHM):
       acov19DAG      daily reported cases by region
       ecov19ivavtid  weekly cases / new ICU / deaths by region
       xcov19ivavDAG  daily new ICU / deaths, national
  3. Daily archive of the FHM Excel (one file per publication day) from
     github.com/adamaltmejd/covid data/FHM/ -- used to difference regional cumulative
     deaths / ICU into daily increments by *report date*.
  4. Svenska Intensivvårdsregistret (SIR) SIRI portal API: new ICU admissions per day
     (report corona_vtfstart), per region of the ICU unit.
  5. Socialstyrelsen deaths (death certificates) and inpatient Excel files (current + Wayback 2022).
  6. SCB population by län (PxWeb BE0101A/BefolkningNy).
"""
import gzip, json, os, sys, time, urllib.request, urllib.parse, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, '..', 'raw')
UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/124.0 Safari/537.36')
FORCE = '--force' in sys.argv


def get(url, path, data=None, headers=None, tries=4):
    path = os.path.join(RAW, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0 and not FORCE:
        return path
    h = {'User-Agent': UA}
    h.update(headers or {})
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=h)
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
            if b[:2] == b'\x1f\x8b':  # Wayback sometimes serves gzip without decoding
                b = gzip.decompress(b)
            with open(path, 'wb') as f:
                f.write(b)
            print('ok ', path, len(b))
            return path
        except Exception as e:  # noqa
            print('retry', k, url, e)
            time.sleep(3 * (k + 1))
    raise RuntimeError('failed ' + url)


# 1. FHM Excel, final archived version
WB = ('https://web.archive.org/web/20230323234341id_/'
      'https://www.arcgis.com/sharing/rest/content/items/b5e7488e117749c19881cce45db13f7e/data')
get(WB, 'Folkhalsomyndigheten_Covid19_wayback20230323.xlsx')

# 2. FHM Folkhälsodata PxWeb
PX = ('https://fohm-app.folkhalsomyndigheten.se/Folkhalsodata/api/v1/sv/A_Folkhalsodata/'
      'H_Sminet/Covid19/falldata/')
q_all = json.dumps({'query': [], 'response': {'format': 'json'}}).encode()
for t in ['acov19DAG.px', 'ecov19ivavtid.px', 'xcov19ivavDAG.px']:
    get(PX + t, f'fhm_px/{t}.meta.json')
    get(PX + t, f'fhm_px/{t}.data.json', data=q_all, headers={'Content-Type': 'application/json'})

# 3. adamaltmejd/covid daily archive of FHM Excel (2020 publication days)
AA = 'https://raw.githubusercontent.com/adamaltmejd/covid/master/data/FHM/'
listing = get('https://api.github.com/repos/adamaltmejd/covid/contents/data/FHM',
              'fhm_archive/_listing.json')
names = [x['name'] for x in json.load(open(listing))]
for n in names:
    if n.startswith('Folkhalsomyndigheten_Covid19_2020-') or n.startswith('Folkhalsomyndigheten_Covid19_2021-01-0'):
        get(AA + n, 'fhm_archive/' + n)

# 4. SIR: ICU admissions per day, per region (ICU unit's region)
SIRP = 'https://portal.icuregswe.org/siri/api/params?data=ltavd&gettree=true&format=json'
tree = json.load(open(get(SIRP, 'sir/sir_units_by_region.json')))[0]['Children']
SIRR = 'https://portal.icuregswe.org/siri/api/reports/generateHighChart'


def sir(avd, fname):
    p = [('language', 'sv'), ('startdat', '2020-01-01'), ('stopdat', '2020-12-31'),
         ('reportName', 'corona_vtfstart'), ('chartWidth', '900'), ('sasong[0]', '2020')]
    p += [(f'avd[{i}]', a) for i, a in enumerate(avd)]
    get(SIRR, 'sir/' + fname, data=urllib.parse.urlencode(p).encode(),
        headers={'Content-Type': 'application/x-www-form-urlencoded'})


sir([], 'vtfstart_riket.json')
for reg in tree:
    sir([c['Key'] for c in reg['Children']], f"vtfstart_{reg['Value'].replace(' ', '').replace('.', '')}.json")
# also the alphabetic unit list, to detect units not assigned to a region in the tree
get('https://portal.icuregswe.org/siri/api/params?data=avdelningar&gettree=true&format=json',
    'sir/sir_units_alphabetic.json')

# 5. Socialstyrelsen
SOS = 'https://www.socialstyrelsen.se/globalassets/sharepoint-dokument/dokument-webb/statistik/'
get(SOS + 'statistik-covid19-avlidna.xlsx', 'sos_statistik-covid19-avlidna.xlsx')
get(SOS + 'statistik-covid19-inskrivna.xlsx', 'sos_statistik-covid19-inskrivna.xlsx')
get('https://web.archive.org/web/20220714133803id_/' + SOS + 'statistik-covid19-avlidna.xlsx',
    'sos_avlidna_wayback20220714.xlsx')

# 6. SCB population by län, 31 Dec 2019 and 31 Dec 2020
SCB = 'https://api.scb.se/OV0104/v1/doris/sv/ssd/START/BE/BE0101/BE0101A/BefolkningNy'
lan = ['01', '03', '04', '05', '06', '07', '08', '09', '10', '12', '13', '14', '17', '18', '19',
       '20', '21', '22', '23', '24', '25']
q = {'query': [{'code': 'Region', 'selection': {'filter': 'vs:RegionLän07', 'values': lan}},
               {'code': 'ContentsCode', 'selection': {'filter': 'item', 'values': ['BE0101N1']}},
               {'code': 'Tid', 'selection': {'filter': 'item', 'values': ['2019', '2020']}}],
     'response': {'format': 'json'}}
get(SCB, 'scb_population_lan_2019_2020.json', data=json.dumps(q).encode(),
    headers={'Content-Type': 'application/json'})
print('done', datetime.datetime.now())

# 7. Definition documents (kept for the record; the FHM covid data-source page was moved/removed, Wayback copy)
get('https://web.archive.org/web/20250914185521id_/https://www.folkhalsomyndigheten.se/folkhalsorapportering-statistik/'
    'statistik-a-o/sjukdomsstatistik/covid-19-veckorapporter/beskrivning-av-datakallor-for-overvakning-av-covid-19/',
    'docs/fhm_beskrivning_datakallor_covid19.html')
get('https://www.folkhalsomyndigheten.se/globalassets/statistik-uppfoljning/smittsamma-sjukdomar/'
    'veckorapporter-covid-19/2020/overvakningssystem-for-covid-19-v1.pdf',
    'docs/fhm_overvakningssystem_for_covid-19_v1_2020.pdf')
get('https://www.socialstyrelsen.se/statistik-och-data/statistik/alla-statistikamnen/'
    'lagesbild-covid-19-influensa-och-rs-statistik/tidigare-publicerad-statistik/datakallor-for-avlidna-i-covid-19/',
    'docs/sos_datakallor_for_avlidna_i_covid-19.html')
