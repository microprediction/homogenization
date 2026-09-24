"""Write tools/pages/map.html: the literature map, built from tools/biblio.json.

Regime-switching homogenization papers sit inside the hub circle; the strands around it are the classical
averaging theory, regime-switching models and bond prices, fast stochastic volatility, and switching in physics
and biology. Node summaries are the bibliography annotations. Run python3 tools/make_map.py, then assemble.py."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIB = {e["key"]: e for e in json.loads((ROOT / "tools" / "biblio.json").read_text())}
CX, CY, R = 500, 335, 178

# key, label, strand, x, y
NODES = [
    ("Cotton2001", "Cotton 2001 (thesis)", "hub", 500, 245),
    ("OURS2026", "Cotton 2026: all orders", "hub", 490, 420),
    ("YinZhang1998", "Yin & Zhang 1998", "core", 385, 285),
    ("IlinKhasminskiiYin1999", "Il'in et al. 1999", "core", 390, 370),
    ("Yin2009", "Yin 2009", "core", 590, 300),
    ("BasuGhosh2009", "Basu & Ghosh 2009", "core", 585, 370),
    ("HuangMandjesSpreij2014", "Huang et al. 2014", "core", 400, 450),
    ("HuangEtAl2016", "Huang et al. 2016", "core", 560, 465),
    ("LawleyKeener2015", "Lawley & Keener 2015", "core", 410, 205),
    ("MonmarcheStrickler2025", "Monmarché & Strickler 2025", "core", 560, 205),
    ("Khasminskii1966", "Khas'minskii 1966", "classic", 110, 70),
    ("GriegoHersh1969", "Griego & Hersh 1969", "classic", 240, 40),
    ("Kurtz1973", "Kurtz 1973", "classic", 380, 60),
    ("PSV1977", "Papanicolaou et al. 1977", "classic", 520, 40),
    ("BLP1978", "Bensoussan et al. 1978", "classic", 690, 70),
    ("PavliotisStuart2008", "Pavliotis & Stuart 2008", "classic", 820, 40),
    ("KhasminskiiYin1996", "Khasminskii & Yin 1996", "classic", 180, 150),
    ("Hamilton1989", "Hamilton 1989", "models", 880, 130),
    ("Landen2000", "Landén 2000", "models", 800, 190),
    ("ElliottMamon2002", "Elliott & Mamon 2002", "models", 760, 280),
    ("ElliottSiu2009", "Elliott & Siu 2009", "models", 870, 240),
    ("vanBeekEtAl2020", "van Beek et al. 2020", "models", 880, 340),
    ("RodrigoMamon2021", "Rodrigo & Mamon 2021", "models", 800, 400),
    ("PapanicolaouSircar2014", "Papanicolaou & Sircar 2014", "models", 760, 480),
    ("Vasicek1977", "Vasicek 1977", "models", 880, 460),
    ("SircarPapanicolaou1999", "Sircar & Papanicolaou 1999", "sv", 150, 560),
    ("FPS2000", "Fouque et al. 2000", "sv", 300, 600),
    ("FPSS2003SIAP", "Fouque et al. 2003", "sv", 170, 630),
    ("CFPS2004", "Cotton et al. 2004", "sv", 440, 590),
    ("DeSantiagoFouqueSolna2008", "DeSantiago et al. 2008", "sv", 580, 630),
    ("FPSS2011", "Fouque et al. 2011", "sv", 680, 580),
    ("Anderson1954", "Anderson 1954", "science", 60, 270),
    ("Kubo1954", "Kubo 1954", "science", 110, 330),
    ("FredkinRice1986", "Fredkin & Rice 1986", "science", 70, 410),
    ("Bressloff2017", "Bressloff 2017", "science", 170, 470),
]
OURS = dict(title="Survival Under Regime-Switching Ornstein–Uhlenbeck Hazard Rates: The Fast-Switching Expansion to All Orders",
            ann="Carries the thesis expansion to all orders in the inverse switching rate, including the initial layer, "
                "checked against a numerical solution of the two-state system.",
            url="./regime-switching.html")
LINKS = [
    ("Cotton2001", "OURS2026", "origin"),
    ("YinZhang1998", "Yin2009", "chain expansions"),
    ("IlinKhasminskiiYin1999", "Yin2009", "densities to option prices"),
    ("KhasminskiiYin1996", "IlinKhasminskiiYin1999", "asymptotic series"),
    ("YinZhang1998", "IlinKhasminskiiYin1999", "two-time-scale chains"),
    ("BasuGhosh2009", "Yin2009", "fast and slow limits"),
    ("HuangMandjesSpreij2014", "HuangEtAl2016", "weak limit"),
    ("Khasminskii1966", "PSV1977", "averaging"),
    ("Khasminskii1966", "KhasminskiiYin1996", "averaging"),
    ("GriegoHersh1969", "Kurtz1973", "random evolutions"),
    ("Kurtz1973", "PSV1977", "random evolutions"),
    ("PSV1977", "BLP1978", "probabilistic side"),
    ("PavliotisStuart2008", "BLP1978", "textbook"),
    ("BLP1978", "OURS2026", "corrector method"),
    ("Hamilton1989", "Landen2000", "regime models"),
    ("Landen2000", "ElliottSiu2009", "semi-affine precursor"),
    ("ElliottSiu2009", "ElliottMamon2002", "generalizes"),
    ("vanBeekEtAl2020", "ElliottSiu2009", "affine on the enlarged state"),
    ("RodrigoMamon2021", "ElliottSiu2009", "time-dependent coefficients"),
    ("ElliottMamon2002", "Cotton2001", "exact bond price, equal volatilities"),
    ("Vasicek1977", "Cotton2001", "the model in each regime"),
    ("PapanicolaouSircar2014", "Yin2009", "slow regime shifts, the opposite limit"),
    ("SircarPapanicolaou1999", "FPS2000", "fast volatility in option pricing"),
    ("FPSS2003SIAP", "FPS2000", "error analysis"),
    ("FPS2000", "CFPS2004", "fast-scale template"),
    ("CFPS2004", "FPSS2011", "collected in"),
    ("DeSantiagoFouqueSolna2008", "FPSS2011", "bond corrections"),
    ("CFPS2004", "OURS2026", "chain replaces diffusion"),
    ("Anderson1954", "Kubo1954", "the Kubo–Anderson model"),
]
STRANDS = {
    "classic": ("#9a9a9a", "Averaging and homogenization theory"),
    "models": ("#0f8b6e", "Regime-switching models and bond prices"),
    "sv": ("#ef6c00", "Fast stochastic volatility"),
    "science": ("#7a5c9e", "Switching in physics and biology"),
}


def node(key, label, strand, x, y):
    if key == "OURS2026":
        e = OURS
        return dict(id=key, label=label, strand=strand, x=x, y=y, **e)
    b = BIB[key]
    return dict(id=key, label=label, strand=strand, x=x, y=y, title=b["title"], ann=b["annotation"],
                url=b.get("url") or ("https://doi.org/" + b["doi"]))


ORDER = ["hub", "core", "classic", "models", "sv", "science"]


def year(label):
    return int(re.search(r"(\d{4})", label).group(1))


def main():
    nodes = [node(*n) for n in NODES]
    # DOM order is the Tab order: the hub, then the works inside the circle, then each strand, each in year order
    nodes.sort(key=lambda n: (ORDER.index(n["strand"]), year(n["label"]), n["label"]))
    label = {n["id"]: n["label"] for n in nodes}
    for n in nodes:
        n["rel"] = [dict(id=b if a == n["id"] else a, label=l) for a, b, l in LINKS if n["id"] in (a, b)]
    data = dict(nodes=nodes, links=[dict(source=a, target=b, label=l) for a, b, l in LINKS])
    assert all(l["source"] in label and l["target"] in label for l in data["links"])
    rows = "\n".join(f"      <li>{label[a]} &ndash; {label[b]}: {l}</li>" for a, b, l in LINKS)
    page = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__RELATIONS__", rows) \
        .replace("__STRANDS__", json.dumps(STRANDS)).replace("__CX__", str(CX)).replace("__CY__", str(CY)).replace("__R__", str(R))
    (ROOT / "tools" / "pages" / "map.html").write_text(page)
    print("wrote tools/pages/map.html:", len(data["nodes"]), "nodes,", len(data["links"]), "links")


TEMPLATE = """<!-- title: Literature map · homogenization -->
  <style>
    #mapwrap { position: relative; border: 1px solid #e2e2e2; border-radius: 8px; background: #fff; overflow-x: auto; }
    #map { display: block; width: 100%; min-width: 960px; height: auto; }
    #map a { outline: none; cursor: pointer; }
    #map a:focus-visible .mark { stroke: #1a1a1a; stroke-width: 2.5; }
    #map a:focus-visible text { text-decoration: underline; }
    #tip { position: absolute; display: none; box-sizing: border-box; max-width: 340px; background: #fff; border: 1px solid #e2e2e2;
      border-radius: 6px; padding: 8px 10px; font-size: 13px; line-height: 1.45; box-shadow: 0 2px 8px rgba(0,0,0,.08); pointer-events: none; z-index: 5; }
    #tip .rel { margin: 4px 0 0; padding-left: 1.1em; color: #5a5a5a; }
    #relations { margin-top: 12px; font-size: .92rem; }
    #relations ul { columns: 2; column-gap: 32px; padding-left: 1.2em; margin: 8px 0 0; }
    #relations li { break-inside: avoid; margin-bottom: 3px; }
    @media (max-width: 640px) { #relations ul { columns: 1; } }
  </style>
  <main style="max-width:1100px">
    <h1>Literature map</h1>
    <p class="subtitle">Regime-switching homogenization at the centre, surrounded by the averaging theory it rests on, the regime-switching models it expands, fast stochastic volatility, and switching in physics and biology.</p>
    <p>The circle holds the works that expand a quantity around its fast-switching average, for a diffusion or
    an ordinary differential equation whose coefficients follow a fast Markov chain.</p>
    <p>Hover over, tap or tab to a node for its summary and the relations it takes part in; click it or press Enter
    to open the work. Hover over an edge for its relation, or read the relations as a list below the map. Drag
    nodes to untangle. On a narrow screen the map scrolls sideways. The full citations are in the
    <a href="./bibliography.html">bibliography</a>, and the <a href="./timeline.html">timeline</a> places the same works in time.</p>
    <div id="legend" style="display:flex;flex-wrap:wrap;gap:18px;font-size:.88rem;color:#5a5a5a;margin:10px 0"></div>
    <div id="mapwrap">
      <svg id="map" viewBox="0 0 1000 680" role="group" aria-label="Literature map: works as nodes, relations as edges"></svg>
    </div>
    <details id="relations">
      <summary>The relations as a list</summary>
      <ul>
__RELATIONS__
      </ul>
    </details>
  </main>
  <div id="tip" role="tooltip"></div>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script>
  (function () {
    var D = __DATA__;
    var S = __STRANDS__, CX = __CX__, CY = __CY__, R = __R__;
    var HUB = '#1f1d3d', CORE = '#4a3aff';
    function col(n) { return n.strand === 'hub' ? HUB : n.strand === 'core' ? CORE : S[n.strand][0]; }
    var inside = function (n) { return n.strand === 'hub' || n.strand === 'core'; };
    var byId = {}; D.nodes.forEach(function (n) { byId[n.id] = n; });
    var strandName = function (n) { return inside(n) ? 'regime-switching homogenization' : S[n.strand][1].toLowerCase(); };
    d3.select('#legend').html('<span><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="6" fill="#ecebfb" stroke="' + CORE + '"/></svg> Regime-switching homogenization</span>' +
      Object.keys(S).map(function (k) { return '<span><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="5" fill="' + S[k][0] + '"/></svg> ' + S[k][1] + '</span>'; }).join(''));
    var svg = d3.select('#map'), tip = d3.select('#tip'), wrap = document.getElementById('mapwrap');
    var current = null, touched = false;
    D.nodes.forEach(function (n) { n.fx0 = n.x; n.fy0 = n.y; });
    var deco = svg.append('g').attr('aria-hidden', 'true');
    deco.append('circle').attr('cx', CX).attr('cy', CY).attr('r', R).attr('fill', '#ecebfb').attr('stroke', CORE).attr('stroke-width', 1.2);
    deco.append('text').attr('x', CX).attr('y', CY - R - 10).attr('text-anchor', 'middle').attr('font-size', 13).attr('font-weight', 600)
      .attr('fill', CORE).text('Regime-switching homogenization');
    var link = deco.append('g').selectAll('line').data(D.links).join('line').attr('stroke', '#c9c9c9').attr('stroke-width', 1.4);
    var hit = deco.append('g').selectAll('line').data(D.links).join('line').attr('stroke', 'transparent').attr('stroke-width', 12)
      .on('mousemove', function (ev, l) { show(l, '<i>' + l.source.label + ' &ndash; ' + l.target.label + ': ' + l.label + '</i>', ev.pageX, ev.pageY); })
      .on('mouseleave', hide);
    function html(n) {
      return '<b style="font-weight:600">' + n.label + '</b><br><i>' + n.title + '</i><br>' + n.ann +
        (n.rel.length ? '<ul class="rel">' + n.rel.map(function (r) { return '<li>' + byId[r.id].label + ': ' + r.label + '</li>'; }).join('') + '</ul>' : '');
    }
    var node = svg.append('g').selectAll('a').data(D.nodes).join('a')
      .attr('href', function (n) { return n.url; })
      .attr('target', function (n) { return n.url.charAt(0) === '.' ? null : '_blank'; })
      .attr('rel', function (n) { return n.url.charAt(0) === '.' ? null : 'noopener'; })
      .attr('draggable', 'false')
      .attr('aria-label', function (n) { return n.label + ', ' + strandName(n); })
      .attr('aria-describedby', 'tip')
      .on('pointerdown', function (ev) { touched = ev.pointerType !== 'mouse'; })
      .on('keydown', function () { touched = false; })
      // on touch the first tap shows the summary and the second opens the work
      .on('click', function (ev, n) { if (touched && current !== n) { ev.preventDefault(); var b = this.querySelector('.mark').getBoundingClientRect();
        show(n, html(n), b.left + window.scrollX + b.width / 2, b.bottom + window.scrollY); } })
      .on('mousemove', function (ev, n) { show(n, html(n), ev.pageX, ev.pageY); })
      .on('mouseleave', hide)
      .on('focus', function (ev, n) { reveal(this); showNode(this, n); })
      .on('blur', hide)
      .call(d3.drag().on('start', function (ev, n) { if (!ev.active) sim.alphaTarget(0.2).restart(); n.fx = n.x; n.fy = n.y; })
        .on('drag', function (ev, n) { n.fx = ev.x; n.fy = ev.y; }).on('end', function (ev, n) { if (!ev.active) sim.alphaTarget(0); n.fx = null; n.fy = null; n.fx0 = n.x; n.fy0 = n.y; }));
    node.append('circle').attr('class', 'mark').attr('r', function (n) { return n.strand === 'hub' ? 9 : 6.5; }).attr('fill', col).attr('stroke', '#fff').attr('stroke-width', 2);
    node.append('text').attr('aria-hidden', 'true').attr('x', 11).attr('y', 4).attr('font-size', function (n) { return inside(n) ? 12.5 : 11.5; })
      .attr('font-weight', function (n) { return n.strand === 'hub' ? 600 : 400; }).attr('fill', '#1a1a1a').text(function (n) { return n.label; });
    var sim = d3.forceSimulation(D.nodes)
      .force('link', d3.forceLink(D.links).id(function (n) { return n.id; }).strength(0.02))
      .force('x', d3.forceX(function (n) { return n.fx0; }).strength(0.35)).force('y', d3.forceY(function (n) { return n.fy0; }).strength(0.35))
      .force('collide', d3.forceCollide(30)).alpha(0.6).on('tick', tick);
    function tick() {
      D.nodes.forEach(function (n) {
        var dx = n.x - CX, dy = n.y - CY, d = Math.sqrt(dx * dx + dy * dy) || 1;
        if (inside(n) && d > R - 22) { n.x = CX + dx * (R - 22) / d; n.y = CY + dy * (R - 22) / d; }
        if (!inside(n) && d < R + 16) { n.x = CX + dx * (R + 16) / d; n.y = CY + dy * (R + 16) / d; }
        n.x = Math.max(12, Math.min(860, n.x)); n.y = Math.max(14, Math.min(666, n.y));
      });
      [link, hit].forEach(function (s) { s.attr('x1', function (l) { return l.source.x; }).attr('y1', function (l) { return l.source.y; }).attr('x2', function (l) { return l.target.x; }).attr('y2', function (l) { return l.target.y; }); });
      node.attr('transform', function (n) { return 'translate(' + n.x + ',' + n.y + ')'; });
    }
    // a focused node and its label are brought inside the scrolling box, since the browser only scrolls for nodes wholly hidden
    function reveal(a) {
      var r = a.getBoundingClientRect(), w = wrap.getBoundingClientRect();
      if (r.left < w.left + 12) wrap.scrollLeft -= w.left + 12 - r.left;
      else if (r.right > w.right - 12) wrap.scrollLeft += Math.min(r.right - w.right + 12, r.left - w.left - 12);
    }
    function showNode(a, n) { var b = a.querySelector('.mark').getBoundingClientRect(); show(n, html(n), b.right + window.scrollX, b.bottom + window.scrollY); }
    // the tooltip lives outside the scrolling box, in page coordinates, and is kept inside the viewport
    function show(d, h, px, py) {
      current = d;
      tip.html(h).style('display', 'block').style('left', '0px').style('top', '0px');
      var t = tip.node(), vw = document.documentElement.clientWidth;
      t.style.maxWidth = Math.min(340, vw - 16) + 'px';
      var w = t.offsetWidth, hh = t.offsetHeight, left = px + 14, top = py + 12;
      if (left + w > window.scrollX + vw - 8) left = Math.max(window.scrollX + 8, px - 14 - w);
      if (top + hh > window.scrollY + window.innerHeight - 8) top = Math.max(window.scrollY + 8, py - 12 - hh);
      tip.style('left', left + 'px').style('top', top + 'px');
    }
    function hide() { current = null; tip.style('display', 'none'); }
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') hide(); });
    document.addEventListener('pointerdown', function (e) { if (!e.target.closest || !e.target.closest('#map a')) hide(); });
    wrap.addEventListener('scroll', function () { var a = document.activeElement; if (current && a && a.closest && a.closest('#map a')) showNode(a, d3.select(a).datum()); });
  })();
  </script>
"""

if __name__ == "__main__":
    main()
