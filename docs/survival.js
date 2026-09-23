// Survival under a regime-switching Ornstein-Uhlenbeck hazard, shared by the demo pages.
// dx = k (th_y - x) dt + sig_y dW, y switches between two regimes at rate lam each way,
// u_y(t, x) = E[exp(-int_0^t x) | x_0 = x, y_0 = y]. Regime 0 is the one with thetas[0].
// Reference implementation: papers/regime-switching-survival/{exact,expansion}.py
(function (root) {
  function make(p) {
    var k = p.kappa, lam = p.lmbd;
    var thb = (p.thetas[0] + p.thetas[1]) / 2, tht = (p.thetas[0] - p.thetas[1]) / 2;
    var s0 = p.sigmas[0] * p.sigmas[0], s1 = p.sigmas[1] * p.sigmas[1];
    var ssb = (s0 + s1) / 2, sst = (s0 - s1) / 2;
    function B(r) { return (1 - Math.exp(-k * r)) / k; }
    function gbar(r) { var b = B(r); return -k * thb * b + 0.5 * ssb * b * b; }
    function gtil(r) { var b = B(r); return -k * tht * b + 0.5 * sst * b * b; }
    function gtilP(r) { var b = B(r), bp = Math.exp(-k * r); return -k * tht * bp + sst * b * bp; }
    function simpson(f, t, n) {
      n = n || Math.max(2000, Math.ceil(t * 2000)); if (n % 2) n++;
      var h = t / n, s = f(0) + f(t);
      for (var i = 1; i < n; i++) s += (i % 2 ? 4 : 2) * f(i * h);
      return s * h / 3;
    }
    // Exact: u_y = a_y(t) exp(-B x), a' = diag(-k th_y B + sig_y^2 B^2 / 2) a + Q a, a(0) = 1.
    function exactA(t, n) {
      n = n || Math.max(400, Math.ceil(t * Math.max(40, 8 * lam)));
      var h = t / n, a0 = 1, a1 = 1, th0 = p.thetas[0], th1 = p.thetas[1];
      function f(r, x0, x1) {
        var b = B(r);
        return [(-k * th0 * b + 0.5 * s0 * b * b) * x0 + lam * (x1 - x0),
                (-k * th1 * b + 0.5 * s1 * b * b) * x1 + lam * (x0 - x1)];
      }
      for (var i = 0; i < n; i++) {
        var r = i * h;
        var q1 = f(r, a0, a1);
        var q2 = f(r + h / 2, a0 + h / 2 * q1[0], a1 + h / 2 * q1[1]);
        var q3 = f(r + h / 2, a0 + h / 2 * q2[0], a1 + h / 2 * q2[1]);
        var q4 = f(r + h, a0 + h * q3[0], a1 + h * q3[1]);
        a0 += h / 6 * (q1[0] + 2 * q2[0] + 2 * q3[0] + q4[0]);
        a1 += h / 6 * (q1[1] + 2 * q2[1] + 2 * q3[1] + q4[1]);
      }
      return [a0, a1];
    }
    function exact(t, x, y) { return exactA(t)[y] * Math.exp(-B(t) * x); }
    // Exact curve on ts = [0, dt, 2dt, ...] in one RK4 pass (sub-steps per grid interval).
    function exactCurve(tmax, npts, x, y) {
      var dt = tmax / (npts - 1), sub = Math.max(8, Math.ceil(dt * Math.max(40, 8 * lam))), h = dt / sub;
      var a0 = 1, a1 = 1, th0 = p.thetas[0], th1 = p.thetas[1], out = [[0, Math.exp(0)]];
      function f(r, x0, x1) {
        var b = B(r);
        return [(-k * th0 * b + 0.5 * s0 * b * b) * x0 + lam * (x1 - x0),
                (-k * th1 * b + 0.5 * s1 * b * b) * x1 + lam * (x0 - x1)];
      }
      for (var i = 1; i < npts; i++) {
        for (var j = 0; j < sub; j++) {
          var r = (i - 1) * dt + j * h;
          var q1 = f(r, a0, a1), q2 = f(r + h / 2, a0 + h / 2 * q1[0], a1 + h / 2 * q1[1]);
          var q3 = f(r + h / 2, a0 + h / 2 * q2[0], a1 + h / 2 * q2[1]), q4 = f(r + h, a0 + h * q3[0], a1 + h * q3[1]);
          a0 += h / 6 * (q1[0] + 2 * q2[0] + 2 * q3[0] + q4[0]); a1 += h / 6 * (q1[1] + 2 * q2[1] + 2 * q3[1] + q4[1]);
        }
        var t = i * dt; out.push([t, (y === 0 ? a0 : a1) * Math.exp(-B(t) * x)]);
      }
      return out;
    }
    // Expansion in eps = 1/lam: returns [averaged, first order, second order, exponentiated].
    function expansion(t, x, y) {
      var e = 1 / lam, sg = y === 0 ? 1 : -1;
      var G = simpson(gbar, t), I = simpson(function (r) { var g = gtil(r); return g * g; }, t);
      var g = gtil(t), gp = gtilP(t), u0 = Math.exp(G - B(t) * x);
      var first = u0 * (1 + e * I / 2 + sg * e * g / 2);
      var second = first + u0 * e * e * (I * I / 8 - g * g / 8 + sg * (I * g / 4 - gp / 4));
      var expo = Math.exp(G - B(t) * x + e * I / 2 - e * e * g * g / 8) * (1 + sg * (e * g / 2 - e * e * gp / 4));
      return [u0, first, second, expo];
    }
    return { B: B, exact: exact, exactCurve: exactCurve, expansion: expansion };
  }
  root.Survival = { make: make };
  if (typeof module !== 'undefined') module.exports = root.Survival;
})(typeof window !== 'undefined' ? window : globalThis);
