// Minimal line chart on canvas: hairline grid, optional log axes, HTML legend, hover crosshair + tooltip.
// Series: {name, color, dash:[...], width, points:[[x,y],...]}. Light-only, per the site style.
(function (root) {
  var INK = '#1a1a1a', MUTED = '#5a5a5a', GRID = '#ececec', AXIS = '#cfcfcf';
  function niceTicks(lo, hi, n) {
    var span = hi - lo, step = Math.pow(10, Math.floor(Math.log10(span / n))), err = span / n / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var t = [], v = Math.ceil(lo / step) * step;
    for (; v <= hi + 1e-12 * span; v += step) t.push(Math.abs(v) < 1e-9 * step ? 0 : +v.toPrecision(12));
    return t;
  }
  function logTicks(lo, hi) {
    var t = [];
    for (var e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      var v = Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) t.push(v);
    }
    return t;
  }
  function fmt(v) {
    var a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1e4 || a < 1e-3) { var e = Math.floor(Math.log10(a)); return (v / Math.pow(10, e)).toFixed(0) === '1' ? '1e' + e : (v / Math.pow(10, e)).toFixed(1) + 'e' + e; }
    return +v.toPrecision(3) + '';
  }
  function LineChart(canvas, opts) {
    this.c = canvas; this.o = opts || {};
    this.tip = document.createElement('div');
    this.tip.style.cssText = 'position:absolute;pointer-events:none;background:#fff;border:1px solid #e2e2e2;border-radius:6px;' +
      'padding:6px 9px;font-size:12.5px;line-height:1.45;box-shadow:0 2px 8px rgba(0,0,0,.08);display:none;z-index:5;white-space:nowrap';
    canvas.parentNode.style.position = 'relative'; canvas.parentNode.appendChild(this.tip);
    var self = this;
    canvas.addEventListener('mousemove', function (e) { self.hover(e); });
    canvas.addEventListener('mouseleave', function () { self.hx = null; self.tip.style.display = 'none'; self.draw(); });
  }
  LineChart.prototype.set = function (series) { this.s = series; this.legend(); this.draw(); };
  LineChart.prototype.legend = function () {
    var id = this.o.legend; if (!id) return;
    var el = document.getElementById(id);
    el.innerHTML = this.s.map(function (s) {
      var d = s.dash && s.dash.length ? 'stroke-dasharray="' + s.dash.join(' ') + '"' : '';
      return '<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px;font-size:.88rem;color:' + MUTED + '">' +
        '<svg width="26" height="8"><line x1="1" y1="4" x2="25" y2="4" stroke="' + s.color + '" stroke-width="2" ' + d + '/></svg>' + s.name + '</span>';
    }).join('');
  };
  LineChart.prototype.geom = function () {
    var dpr = window.devicePixelRatio || 1, W = this.c.clientWidth || this.c.width, H = Math.round(W * this.c.height / this.c.width);
    if (this.c._w !== W * dpr) { this.c.width = W * dpr; this.c.height = H * dpr; this.c._w = W * dpr; this.c.style.height = H + 'px'; }
    var pad = { l: 64, r: 14, t: 12, b: 42 }, o = this.o;
    var xs = [], ys = [];
    this.s.forEach(function (s) { s.points.forEach(function (p) { if (isFinite(p[1]) && (!o.ylog || p[1] > 0)) { xs.push(p[0]); ys.push(p[1]); } }); });
    var x0 = o.xmin != null ? o.xmin : Math.min.apply(null, xs), x1 = o.xmax != null ? o.xmax : Math.max.apply(null, xs);
    var y0 = o.ymin != null ? o.ymin : Math.min.apply(null, ys), y1 = o.ymax != null ? o.ymax : Math.max.apply(null, ys);
    if (!o.ylog && o.ymin == null) { var m = (y1 - y0) * 0.06 || 0.01; y0 -= m; y1 += m; }
    var tx = o.xlog ? function (v) { return Math.log10(v); } : function (v) { return v; };
    var ty = o.ylog ? function (v) { return Math.log10(v); } : function (v) { return v; };
    return { dpr: dpr, W: W, H: H, pad: pad, x0: x0, x1: x1, y0: y0, y1: y1,
      X: function (v) { return pad.l + (tx(v) - tx(x0)) / (tx(x1) - tx(x0)) * (W - pad.l - pad.r); },
      Y: function (v) { return H - pad.b - (ty(v) - ty(y0)) / (ty(y1) - ty(y0)) * (H - pad.t - pad.b); },
      Xinv: function (px) { var f = (px - pad.l) / (W - pad.l - pad.r), v = tx(x0) + f * (tx(x1) - tx(x0)); return o.xlog ? Math.pow(10, v) : v; } };
  };
  LineChart.prototype.draw = function () {
    if (!this.s) return;
    var g = this.geom(), ctx = this.c.getContext('2d'), o = this.o;
    ctx.setTransform(g.dpr, 0, 0, g.dpr, 0, 0); ctx.clearRect(0, 0, g.W, g.H);
    ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    var xt = o.xlog ? logTicks(g.x0, g.x1) : niceTicks(g.x0, g.x1, 6), yt = o.ylog ? logTicks(g.y0, g.y1) : niceTicks(g.y0, g.y1, 5);
    ctx.lineWidth = 1; ctx.strokeStyle = GRID; ctx.fillStyle = MUTED;
    yt.forEach(function (v) { var y = Math.round(g.Y(v)) + .5; ctx.beginPath(); ctx.moveTo(g.pad.l, y); ctx.lineTo(g.W - g.pad.r, y); ctx.stroke();
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(fmt(v), g.pad.l - 8, y); });
    xt.forEach(function (v) { var x = Math.round(g.X(v)) + .5; ctx.beginPath(); ctx.moveTo(x, g.pad.t); ctx.lineTo(x, g.H - g.pad.b); ctx.stroke();
      ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(fmt(v), x, g.H - g.pad.b + 6); });
    ctx.strokeStyle = AXIS; ctx.beginPath(); ctx.moveTo(g.pad.l + .5, g.pad.t); ctx.lineTo(g.pad.l + .5, g.H - g.pad.b + .5); ctx.lineTo(g.W - g.pad.r, g.H - g.pad.b + .5); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    if (o.xlabel) ctx.fillText(o.xlabel, g.pad.l + (g.W - g.pad.l - g.pad.r) / 2, g.H - 4);
    if (o.ylabel) { ctx.save(); ctx.translate(14, g.pad.t + (g.H - g.pad.t - g.pad.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText(o.ylabel, 0, 0); ctx.restore(); }
    ctx.save(); ctx.beginPath(); ctx.rect(g.pad.l, g.pad.t, g.W - g.pad.l - g.pad.r, g.H - g.pad.t - g.pad.b); ctx.clip();
    this.s.forEach(function (s) {
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2; ctx.setLineDash(s.dash || []); ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.beginPath(); var started = false;
      s.points.forEach(function (p) { var ok = isFinite(p[1]) && (!o.ylog || p[1] > 0);
        if (!ok) { started = false; return; } var x = g.X(p[0]), y = g.Y(p[1]);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y); });
      ctx.stroke();
    });
    ctx.setLineDash([]);
    if (this.hx != null) {
      var hx = this.hx; ctx.strokeStyle = '#b8b8b8'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(g.X(hx) + .5, g.pad.t); ctx.lineTo(g.X(hx) + .5, g.H - g.pad.b); ctx.stroke();
      this.s.forEach(function (s) { var p = nearest(s.points, hx); if (!p || !isFinite(p[1]) || (o.ylog && p[1] <= 0)) return;
        ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(g.X(p[0]), g.Y(p[1]), 5, 0, 7); ctx.fill();
        ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(g.X(p[0]), g.Y(p[1]), 3.5, 0, 7); ctx.fill(); });
    }
    ctx.restore();
  };
  function nearest(pts, x) { var b = null, d = Infinity; pts.forEach(function (p) { var e = Math.abs(p[0] - x); if (e < d) { d = e; b = p; } }); return b; }
  LineChart.prototype.hover = function (e) {
    if (!this.s) return;
    var r = this.c.getBoundingClientRect(), g = this.geom(), px = e.clientX - r.left;
    if (px < g.pad.l || px > g.W - g.pad.r) { this.hx = null; this.tip.style.display = 'none'; this.draw(); return; }
    this.hx = nearest(this.s[0].points, g.Xinv(px))[0]; this.draw();
    var o = this.o, hx = this.hx, fy = o.yfmt || function (v) { return v.toPrecision(5); };
    var rows = this.s.map(function (s) { var p = nearest(s.points, hx);
      return '<div><span style="display:inline-block;width:10px;height:2px;background:' + s.color + ';vertical-align:middle;margin-right:6px"></span>' +
        s.name + ': <b style="color:' + INK + ';font-variant-numeric:tabular-nums">' + (p && isFinite(p[1]) ? fy(p[1]) : '&ndash;') + '</b></div>'; });
    this.tip.innerHTML = '<div style="color:' + MUTED + ';margin-bottom:2px">' + (o.xname || 'x') + ' = ' + (o.xfmt ? o.xfmt(hx) : fmt(hx)) + '</div>' + rows.join('');
    this.tip.style.display = 'block';
    var left = this.c.offsetLeft + g.X(hx) + 12; if (left + this.tip.offsetWidth > this.c.offsetLeft + g.W) left = this.c.offsetLeft + g.X(hx) - this.tip.offsetWidth - 12;
    this.tip.style.left = left + 'px'; this.tip.style.top = (this.c.offsetTop + g.pad.t + 4) + 'px';
  };
  root.LineChart = LineChart;
})(window);
