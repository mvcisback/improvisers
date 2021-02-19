from improvisers.pareto import Interval, Pareto, Point


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

    curve = Pareto(
        points=[p1, p2, p3],
        margin=1e-2,
    )

    assert curve.rationality(0.1) == 10
    assert curve.rationality(0.11) == 8
    assert curve.rationality(0.3) == 8
    assert curve.rationality(1) == 2
    assert curve.rationality(3) == 2

    assert curve.win_prob(0.1) == Interval(0.8, 0.8 + 1e-2)
    assert curve.win_prob(0.11) == Interval(0.6, 0.6 + 1e-2)
    assert curve.win_prob(0.3) == Interval(0.6, 0.6 + 1e-2)
    assert curve.win_prob(1) == Interval(0.1, 0.1 + 1e-2)
