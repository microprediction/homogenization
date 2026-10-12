"""Check page sources before a commit or PR. Usage: python3 tools/check_pages.py [tools/pages/<page>.html ...]

With no arguments every page is checked. Reports, and exits nonzero on:
  - LaTeX commands missing their backslash inside $...$ (e.g. `,qquad`, `{mathrm`), which render as text;
  - paragraphs of more than three sentences (fix with tools/split_paragraphs.py);
  - pages over 800 source lines (move detail to <page>-full.html);
  - duplicate entries on the bibliography page, bibliography keys missing from it, and broken
    ./bibliography.html#Key links.
"""
import json, pathlib, re, subprocess, sys, tempfile, shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGES = ROOT / 'tools' / 'pages'
CMDS = ('qquad|mathrm|operatorname|infty|mathbb|mathcal|mathbf|widehat|widetilde|varepsilon|tfrac|ldots|cdots|'
        'bigl|bigr|Bigl|Bigr|oint|nabla|partial|approx')
EXEMPT = {'papers', 'bibliography', 'survey'}  # list pages, not notes
MAX_LINES = 800


def problems(path):
    t = path.read_text()
    out = []
    for m in re.finditer(r'\$\$.+?\$\$|\$.+?\$', t, re.S):
        for h in re.finditer(r'(?<![\\A-Za-z])(' + CMDS + r')(?![A-Za-z])', m.group(0)):
            line = t[:m.start() + h.start()].count('\n') + 1
            out.append(f'line {line}: "{h.group(1)}" is missing its backslash')
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d) / path.name
        shutil.copy(path, tmp)
        subprocess.run([sys.executable, str(ROOT / 'tools' / 'split_paragraphs.py'), str(tmp)], capture_output=True)
        if tmp.read_text() != t:
            out.append('has paragraphs over three sentences; run tools/split_paragraphs.py on it')
    n = t.count('\n')
    if n > MAX_LINES and not path.stem.endswith('-full') and path.stem not in EXEMPT:
        out.append(f'{n} lines; keep summaries under {MAX_LINES} and put detail in {path.stem}-full.html')
    return out


def bibliography_problems(paths):
    page = (PAGES / 'bibliography.html').read_text()
    ids = re.findall(r'<li id="([^"]+)"', page)
    out = [f'bibliography.html: "{k}" appears more than once' for k in sorted({i for i in ids if ids.count(i) > 1})]
    keys = {e['key'] for e in json.loads((ROOT / 'tools' / 'biblio.json').read_text())}
    out += [f'tools/biblio.json: "{k}" is not on bibliography.html' for k in sorted(keys - set(ids))]
    for p in paths:
        for k in sorted(set(re.findall(r'bibliography\.html#([A-Za-z0-9_\-]+)', p.read_text())) - set(ids)):
            out.append(f'{p.relative_to(ROOT)}: link to bibliography.html#{k}, which does not exist')
    return out


def main(args):
    paths = [pathlib.Path(a).resolve() for a in args] or sorted(PAGES.glob('**/*.html'))
    found = []
    for p in paths:
        found += [f'{p.relative_to(ROOT)}: {x}' for x in problems(p)]
    found += bibliography_problems(paths)
    for x in found:
        print(x)
    print('OK' if not found else f'{len(found)} problem(s)')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
