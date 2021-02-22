import math

from improvisers.pareto2 import Pareto

from pytest import approx


oo = float('inf')


def test_pareto_toy():
    """Tests pareto curve of node 5 in the a = 1/3 toy example."""

    def entropy5(coeff):
        if coeff == oo:
            return math.log(2)

        z = math.exp(coeff / 3)  # magic parameter from a = 1/3 example.
        p30 = 1 / (1 + z)
        p54 = z / (1 + 2 * z)
        p32 = 1 - p30
        p53 = 1 - p54

        entropy3 = -p30 * math.log(p30) -p32 * math.log(p32)
        return p53 * (entropy3 - math.log(p53)) - p54 * math.log(p54)

    def win_prob5(coeff):
        if coeff == oo:
            return 1 / 3
        z = math.exp(coeff / 3)  # magic parameter from a = 1/3 example.
        
        return (1/3)*z*(1/(1 + z))

    def win_prob5_hi(coeff):
        if coeff in (0, oo):
            return win_prob5(coeff)

        return win_prob5(coeff) + 1e-4

    curve = Pareto.build(
        lower_win_prob=win_prob5,
        upper_win_prob=win_prob5_hi,
        entropy=entropy5,
        tol=1e-3,
    )
    assert curve.lower_win_prob(entropy5(oo)) == curve.upper_win_prob(entropy5(oo))

    # Always approximate with higher entropy.
    left, right, p = curve.rationality(0.3)
    assert (1 - p)*entropy5(left) + p*entropy5(right) == approx(0.3)

    # Matching entropy 
    target_entropy = entropy5(1)
    win_prob_lowerbound = curve.lower_win_prob(target_entropy)
    win_prob_actual = win_prob5(1)

    assert win_prob_lowerbound < win_prob_actual
    assert win_prob_lowerbound > win_prob_actual - 1e-2
