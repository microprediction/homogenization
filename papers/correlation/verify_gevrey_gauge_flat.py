#!/usr/bin/env python3
"""Certificate for the Gevrey-gauge Green--Kubo flat constant."""

import math

from scipy.optimize import minimize_scalar

from verify_gauge_flat import chain_data, contracted_gauge_rate


GAUGE_CONSTANT = 4.60437900568596
FLAT_POWER = 2.0
FLAT_SCALE = 1.0


def limiting_constant(moment_power):
    """Homogeneous contraction constant for the requested moment."""
    q = FLAT_POWER
    return (
        (q + 2.0) / (2.0 * q)
        * (q * moment_power * FLAT_SCALE) ** (2.0 / (q + 2.0))
        * GAUGE_CONSTANT ** (-q / (q + 2.0))
    )


def exact_saddle(speed, moment_power, scores, green_kubo):
    """Contract the exact finite-chain rate over the gauge radius."""
    q = FLAT_POWER
    predicted_radius = (
        q * moment_power * FLAT_SCALE * GAUGE_CONSTANT / speed
    ) ** (1.0 / (q + 2.0))

    def objective(log_radius):
        radius = math.exp(log_radius)
        rate, _ = contracted_gauge_rate(scores, green_kubo, radius)
        return moment_power * FLAT_SCALE / radius ** q + speed * rate

    result = minimize_scalar(
        objective,
        bounds=(math.log(predicted_radius) - 2.0,
                math.log(predicted_radius) + 2.0),
        method="bounded",
        options={"xatol": 1e-9},
    )
    return math.exp(float(result.x)), float(result.fun), predicted_radius


def main():
    stationary, scores, green_kubo = chain_data()
    second_constant = limiting_constant(2.0)
    first_constant = limiting_constant(1.0)
    separation_constant = 2.0 * first_constant - second_constant
    exponent = FLAT_POWER / (FLAT_POWER + 2.0)

    rows = []
    for speed in (1e4, 1e6, 1e8, 1e10):
        second_radius, second_cost, predicted_radius = exact_saddle(
            speed, 2.0, scores, green_kubo
        )
        _, first_cost, _ = exact_saddle(
            speed, 1.0, scores, green_kubo
        )
        scale = speed ** exponent
        rows.append((
            speed,
            second_radius,
            second_radius / predicted_radius,
            second_cost / (second_constant * scale),
            (2.0 * first_cost - second_cost)
            / (separation_constant * scale),
        ))

    print("Gevrey-gauge Green--Kubo flat-map certificate")
    print(f"stationary law {stationary}")
    print(f"flat power q {FLAT_POWER:.6f}")
    print(f"flat scale b {FLAT_SCALE:.6f}")
    print(f"gauge constant {GAUGE_CONSTANT:.9f}")
    print(f"predicted fractional exponent {exponent:.9f}")
    print(f"first-moment constant {first_constant:.9f}")
    print(f"variance constant {second_constant:.9f}")
    print(f"mean-separation constant {separation_constant:.9f}")
    for speed, radius, radius_ratio, variance, separation in rows:
        print(
            f"speed {speed:.0e}: radius {radius:.9f}, "
            f"radius ratio {radius_ratio:.9f}, variance {variance:.9f}, "
            f"separation {separation:.9f}"
        )

    ok = True
    ok &= separation_constant > 0.0
    ok &= abs(exponent - 0.5) < 1e-14
    ok &= all(rows[i][3] > rows[i + 1][3] > 1.0
              for i in range(len(rows) - 1))
    ok &= all(rows[i][4] > rows[i + 1][4] > 1.0
              for i in range(len(rows) - 1))
    ok &= abs(rows[-1][2] - 1.0) < 2e-4
    ok &= rows[-1][3] < 1.0002
    ok &= rows[-1][4] < 1.0001
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
