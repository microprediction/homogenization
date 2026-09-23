"""Split paragraphs of more than three sentences in a page source (house style), at sentence boundaries that are
outside inline math and HTML tags. Usage: python3 tools/split_paragraphs.py tools/pages/<page>.html"""
import re, sys


def boundaries(text):
    out, in_math, in_tag, i = [], False, False, 0
    while i < len(text):
        c = text[i]
        if c == '<' and not in_math:
            in_tag = True
        elif c == '>' and in_tag:
            in_tag = False
        elif c == '$' and not in_tag:
            in_math = not in_math
        elif c in '.?' and not in_math and not in_tag:
            m = re.match(r'[.?]\s+(?=[A-Z<$])', text[i:])
            if m and not re.search(r'\b(al|e\.g|i\.e|cf|Fig|Eq|vs|St)$', text[max(0, i - 4):i]):
                out.append(i + 1)
        i += 1
    return out


def split(par):
    b = boundaries(par)
    if len(b) + 1 <= 3:
        return [par]
    n = len(b) + 1
    groups = 2 if n <= 6 else (n + 2) // 3
    per = -(-n // groups)
    cuts = [b[k * per - 1] for k in range(1, groups) if k * per - 1 < len(b)]
    pieces, prev = [], 0
    for c in cuts:
        pieces.append(par[prev:c].strip()); prev = c
    pieces.append(par[prev:].strip())
    return pieces


def main(path):
    t = open(path).read()
    def fix(m):
        attrs, body = m.group(1), m.group(2)
        if '$$' in body:
            return m.group(0)
        parts = split(body)
        return '\n    '.join(f'<p{attrs}>{p}</p>' for p in parts)
    t2 = re.sub(r'<p((?: class="(?:lead|muted)")?)>(.*?)</p>', fix, t, flags=re.S)
    open(path, 'w').write(t2)


if __name__ == '__main__':
    main(sys.argv[1])
