from improvisers.pareto import Point, Interval, ParetoCurve


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


def test_point():
    i1 = Interval(0.1, 1)
    p1 = Point(0.2, 0.3, i1)
    p2 = Point(0.2, 0, i1*2)
    p3 = Point(2, 0.2, i1*3)

    assert p1 == p2
    assert p3 > p1
    assert p3 > p2
    assert p3 > 1


def test_curve():
    i1 = Interval(0.1, 1)
    p1 = Point(0.2, 0.3, i1*2)
    p2 = Point(0.1, 0, i1)
    p3 = Point(2, 3, i1*3)

    curve = ParetoCurve([p1, p2, p3])
    assert curve.rationality(0.1) == 0
    assert curve.rationality(0.11) == 0.3
    assert curve.rationality(0.2) == 0.3
    assert curve.rationality(2) == 3

    assert curve.win_prob(0.1) == i1
    assert curve.win_prob(0.11) == i1 * 2
    assert curve.win_prob(0.2) == i1 * 2
    assert curve.win_prob(2) == i1 * 3
