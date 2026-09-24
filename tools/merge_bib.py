"""Merge page bibliography snippets (tools/pages/_bib/<page>.json) into tools/biblio.json and the bibliography page.

Each snippet is a list of {key, authors, year, title, venue, doi, annotation}. Entries whose key already exists are
skipped. New entries go into a section per snippet, headed by SECTION_TITLES[page], before the fluid-dynamics section."""
import html, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SECTION_TITLES = {
    'clumpy-media': 'Radiative transfer in stochastic media',
    'pulsars': 'Pulsar timing',
    'epidemics': 'Epidemics in random environments',
    'portfolio': 'Portfolio choice with regimes',
    'correlation': 'Correlation regimes and correlation risk',
    'lumping': 'Lumped chains and phase-type durations',
}


def main(pages):
    B = ROOT / 'tools' / 'biblio.json'
    bib = json.loads(B.read_text())
    have = {e['key'] for e in bib}
    page = ROOT / 'tools' / 'pages' / 'bibliography.html'
    t = page.read_text()
    for name in pages:
        snip = json.loads((ROOT / 'tools' / 'pages' / '_bib' / f'{name}.json').read_text())
        lis = []
        for e in snip:
            if e['key'] in have:
                continue
            doi = e.get('doi', '').lower()
            url = e.get('url') or (f'https://doi.org/{doi}' if doi else '')
            bib.append({'role': f'survey_{name}', 'key': e['key'], 'authors': e['authors'], 'year': e['year'],
                        'title': e['title'], 'venue': e['venue'], 'doi': doi, 'url': url, 'annotation': e['annotation']})
            have.add(e['key'])
            link = f' <a href="https://doi.org/{doi}">doi:{doi}</a>.' if doi else (f' <a href="{html.escape(url)}">link</a>.' if url else '')
            lis.append(f'        <li id="{e["key"]}">{html.escape(e["authors"])} ({e["year"]}). <em>{html.escape(e["title"])}</em>. '
                       f'{html.escape(e["venue"])}.{link}<br><span class="muted">{html.escape(e["annotation"])}</span></li>')
        if lis and f'id="{name}-refs"' not in t:
            blk = f'      <h3 id="{name}-refs">{html.escape(SECTION_TITLES[name])}</h3>\n      <ul>\n' + '\n'.join(lis) + '\n      </ul>\n'
            anchor = '      <h3 id="fluid">'
            t = t.replace(anchor, blk + anchor, 1)
        print(name, 'added', len(lis))
    B.write_text(json.dumps(bib, indent=1, ensure_ascii=False))
    page.write_text(t)


if __name__ == '__main__':
    main(sys.argv[1:])
