// All-orders fast-switching expansion (outer series plus initial layer), a port of
// papers/regime-switching-survival/all_orders.py. rho = d/m solves rho' = g(1 - rho^2) - 2 lam rho,
// log m = int (gbar + g rho), u_y = exp(-B x) m (1 +/- rho). Outer terms are polynomials in E = 1 - exp(-k t);
// the layer lives in the span of tau^k exp(-2 j tau), tau = lam t.
(function (root) {
  function padd(a, b) { var n = Math.max(a.length, b.length), r = []; for (var i = 0; i < n; i++) r.push((a[i] || 0) + (b[i] || 0)); return r; }
  function pmul(a, b) { var r = new Array(a.length + b.length - 1).fill(0); for (var i = 0; i < a.length; i++) for (var j = 0; j < b.length; j++) r[i + j] += a[i] * b[j]; return r; }
  function pscale(a, s) { return a.map(function (c) { return c * s; }); }
  function pder(a) { if (a.length < 2) return [0]; var r = []; for (var i = 1; i < a.length; i++) r.push(i * a[i]); return r; }
  function pval(a, x) { var v = 0; for (var i = a.length - 1; i >= 0; i--) v = v * x + a[i]; return v; }
  function fact(n) { var f = 1; for (var i = 2; i <= n; i++) f *= i; return f; }
  function comb(n, k) { return fact(n) / (fact(k) * fact(n - k)); }
  function taylorInE(p, kappa, K) { // Taylor coefficients in r of p(E(r))
    var e = [0]; for (var i = 1; i <= K; i++) e.push(-Math.pow(-kappa, i) / fact(i));
    var out = new Array(K + 1).fill(0), pw = new Array(K + 1).fill(0); pw[0] = 1;
    p.forEach(function (c) { for (var i = 0; i <= K; i++) out[i] += c * pw[i]; pw = pmul(pw, e).slice(0, K + 1); });
    return out;
  }
  // ExpPoly: map "k,j" -> c for c tau^k exp(-2 j tau)
  function ep() { return {}; }
  function epAdd(a, b) { var r = Object.assign({}, a); for (var key in b) r[key] = (r[key] || 0) + b[key]; return r; }
  function epMulPoly(a, coeffs) { var r = {}; for (var key in a) { var kj = key.split(','), k = +kj[0], j = +kj[1];
    coeffs.forEach(function (c, i) { if (c) { var nk = (k + i) + ',' + j; r[nk] = (r[nk] || 0) + a[key] * c; } }); } return r; }
  function epMul(a, b) { var r = {}; for (var ka in a) for (var kb in b) { var x = ka.split(','), y = kb.split(','), nk = (+x[0] + +y[0]) + ',' + (+x[1] + +y[1]); r[nk] = (r[nk] || 0) + a[ka] * b[kb]; } return r; }
  function epVal(a, tau) { var v = 0; for (var key in a) { var kj = key.split(','); v += a[key] * Math.pow(tau, +kj[0]) * Math.exp(-2 * kj[1] * tau); } return v; }
  function epInt(a, T) { var v = 0; for (var key in a) { var kj = key.split(','), k = +kj[0], al = 2 * +kj[1], tail = 0;
    for (var i = 0; i <= k; i++) tail += Math.pow(al * T, i) / fact(i); v += a[key] * fact(k) / Math.pow(al, k + 1) * (1 - Math.exp(-al * T) * tail); } return v; }
  function epSolve(f, y0) { var y = {}; for (var key in f) { var kj = key.split(','), k = +kj[0], j = +kj[1], c = f[key];
      if (j === 1) { var nk = (k + 1) + ',1'; y[nk] = (y[nk] || 0) + c / (k + 1); }
      else { var b = 2 * (j - 1); for (var i = 0; i <= k; i++) { var kk = (k - i) + ',' + j; y[kk] = (y[kk] || 0) - c * fact(k) / fact(k - i) / Math.pow(b, i + 1); } } }
    y['0,1'] = (y['0,1'] || 0) + y0 - epVal(y, 0); return y; }

  function make(p, N) {
    N = N || 6;
    var k = p.kappa, lam = p.lmbd, eps = 1 / lam;
    var thb = (p.thetas[0] + p.thetas[1]) / 2, tht = (p.thetas[0] - p.thetas[1]) / 2;
    var s0 = p.sigmas[0] * p.sigmas[0], s1 = p.sigmas[1] * p.sigmas[1];
    var gbar = [0, -thb, (s0 + s1) / 2 / (2 * k * k)], g = [0, -tht, (s0 - s1) / 2 / (2 * k * k)];
    function ddt(q) { return pmul(pder(q), [k, -k]); }
    var rho = [[0], pscale(g, 0.5)];
    for (var n = 1; n < N; n++) { var conv = [0]; for (var i = 1; i < n; i++) conv = padd(conv, pmul(rho[i], rho[n - i]));
      rho.push(pscale(padd(ddt(rho[n]), pmul(g, conv)), -0.5)); }
    var logOuter = [null]; for (n = 1; n <= N; n++) logOuter.push(pmul(g, rho[n]));
    var K = N + 2, G = taylorInE(g, k, K), R = [null]; for (n = 1; n <= N; n++) R.push(taylorInE(rho[n], k, K));
    var eta = []; for (n = 0; n <= N; n++) eta.push(ep());
    for (n = 2; n <= N; n++) {
      var f = ep();
      for (var a = 1; a < n; a++) for (var b = 1; b < n; b++) for (var c = 0; c < n; c++) {
        var m = n - 1 - a - b - c; if (m >= 2 && G[a] && R[b][c]) { var co = new Array(a + c).fill(0); co.push(-2 * G[a] * R[b][c]); f = epAdd(f, epMulPoly(eta[m], co)); } }
      for (a = 1; a < n; a++) for (var m1 = 2; m1 < n; m1++) { var m2 = n - 1 - a - m1;
        if (m2 >= 2 && G[a]) { var co2 = new Array(a).fill(0); co2.push(-G[a]); f = epAdd(f, epMulPoly(epMul(eta[m1], eta[m2]), co2)); } }
      eta[n] = epSolve(f, -pval(rho[n], 0));
    }
    function J(q, t) { var tot = 0; q.forEach(function (c, n) { if (!c) return; var s = t;
      for (var j = 1; j <= n; j++) s += comb(n, j) * Math.pow(-1, j) * (1 - Math.exp(-j * k * t)) / (j * k); tot += c * s; }); return tot; }
    // partial sums u^(0..N)(t, x, y): order n keeps every term through eps^n
    function partialSums(t, x, y) {
      var E = 1 - Math.exp(-k * t), T = t / eps, sg = y === 0 ? 1 : -1, out = [];
      for (var ord = 0; ord <= N; ord++) {
        var logm = J(gbar, t), r = 0;
        for (var n = 1; n <= ord; n++) { logm += Math.pow(eps, n) * J(logOuter[n], t); r += Math.pow(eps, n) * pval(rho[n], E); }
        for (n = 2; n <= ord; n++) r += Math.pow(eps, n) * epVal(eta[n], T);
        for (var a = 1; a <= ord; a++) for (var m = 2; m <= ord; m++) if (1 + a + m <= ord && G[a]) {
          var co = new Array(a).fill(0); co.push(1); logm += Math.pow(eps, 1 + a + m) * G[a] * epInt(epMulPoly(eta[m], co), T); }
        out.push(Math.exp(-E / k * x + logm) * (1 + sg * r));
      }
      return out;
    }
    return { partialSums: partialSums, order: N };
  }
  root.AllOrders = { make: make };
  if (typeof module !== 'undefined') module.exports = root.AllOrders;
})(typeof window !== 'undefined' ? window : globalThis);
