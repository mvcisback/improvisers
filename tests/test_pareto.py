import math

from improvisers.pareto import Interval, Pareto, Point

from pytest import approx


oo = float('inf')


def test_interval():
    i1 = Interval(0.1, 1.1)
    i2 = Interval(0.3, 4)

    assert 0.2 in i1
    assert 0.2 not in i2
    assert i1 < 2
    assert i1 > 0
    assert not (i1 < 0.5)
    assert not (i1 > 0.5)
    assert i1 != 0.5

    assert i1 + i2 == Interval(0.4, 5.1)
    assert i1 * 2.0 == Interval(0.2, 2.2)

    assert i1.size == 1


def test_curve1():
    p1 = Point(entropy=0.1, rationality=10, win_prob=0.8)
    p2 = Point(entropy=3, rationality=2, win_prob=0.1)
    p3 = Point(entropy=0.3, rationality=8, win_prob=0.6)
    p4 = Point(entropy=0, rationality=float('inf'), win_prob=1)

    curve = Pareto(
        points=[p1, p2, p3],
        margin=1e-2,
    )

    assert curve.rationality(0.1) == 10
    assert curve.rationality(0.11) == 8
    assert curve.rationality(0.3) == 8
    assert curve.rationality(1) == 2
    assert curve.rationality(3) == 2

    assert curve.win_prob(0.1) == Interval(0.8, 0.8)
    assert curve.win_prob(0.11) == Interval(0.6, 0.6 + 1e-2)
    assert curve.win_prob(0.3) == Interval(0.6, 0.6 + 1e-2)
    assert curve.win_prob(1) == Interval(0.1, 0.1 + 1e-2)


def test_pareto_toy():
    """Tests pareto curve of node 5 in the a = 1/3 toy example."""

    def entropy5(coeff):
        if coeff == float('inf'):
            return 0

        z = math.exp(coeff / 3)  # magic parameter from a = 1/3 example.
        p30 = 1 / (1 + z)
        p54 = z / (1 + 2 * z)
        p32 = 1 - p30
        p53 = 1 - p54

        entropy3 = -p30 * math.log(p30) -p32 * math.log(p32)
        return p53 * (entropy3 - math.log(p53)) - p54 * math.log(p54)

    def win_prob5(coeff):
        if coeff == float('inf'):
            return 1 / 3
        z = math.exp(coeff / 3)  # magic parameter from a = 1/3 example.
        
        return z / (1 + z) / 3

    def find_point(coeff):
        return Point(
            rationality=coeff,
            entropy=entropy5(coeff),
            win_prob=win_prob5(coeff),
        )

    curve = Pareto.build(find_point, tol=1e-3, prev_margin=1e-3)
    assert curve.win_prob(entropy5(oo)).size == 0
    assert curve.max_win_prob == 1/3
    assert curve.max_entropy == entropy5(0)

    # Always approximate with higher entropy.
    assert entropy5(curve.rationality(0.3)) >= 0.3

    # Matching entropy 
    target_entropy = entropy5(1)
    win_prob_lowerbound = curve.win_prob(target_entropy).low
    win_prob_actual = win_prob5(1)

    assert win_prob_lowerbound <= win_prob_actual
    assert win_prob_actual - win_prob_lowerbound <= 1e-2
