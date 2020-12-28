from pytest import approx

import math

import improvisers as I


def test_deterministic_critic():
    game_graph = I.ExplicitGameGraph(
        root=0,
        graph={
            0: (False, {}),
            1: (True, {}),
            2: ('p1', {0, 1}),
            3: ('p1', {0, 2}),
            4: ('p1', {0, 3}),
        }
    )
    critic = I.TabularCritic.from_game_graph(game_graph)

    # Test value function.

    assert critic.value(0, 0) == 0
    for i in range(1, 5):
        assert critic.value(i, 0) == approx(math.log(i))

    coeff = math.log(2)
    for i in range(1, 5):
        assert critic.value(i, math.log(2)) == approx(math.log(i + 1))

    # Test action probabilities of policy.
    assert set(critic.action_dist(0, coeff).support()) == set()
    assert set(critic.action_dist(1, coeff).support()) == set()
    assert set(critic.action_dist(2, coeff).support()) == {0, 1}
    assert set(critic.action_dist(3, coeff).support()) == {0, 2}
    assert set(critic.action_dist(4, coeff).support()) == {0, 3}

    for i in range(2, 5):
        dist = critic.action_dist(i, coeff)
        lratio = math.log(dist.prob(i-1)) - math.log(dist.prob(0))
        assert lratio == approx(math.log(i))

    # Test causal entropy of policy.
    assert critic.entropy(0, coeff) == 0
    assert critic.entropy(1, coeff) == 0
    for i in range(2, 5):
        prob1, prob2 = i / (i + 1), 1 / (i + 1)
        delta = critic.entropy(i, coeff) - prob1 * critic.entropy(i-1, coeff)
        assert delta == approx(-prob1*math.log(prob1) - prob2*math.log(prob2))

    # Test psat and rationality are approximate inverses.
    psat = critic.psat(4, coeff)
    coeff2 = critic.rationality(4, psat)
    psat2 = critic.psat(4, coeff2)

    assert approx(psat) == psat2
    assert approx(coeff) == coeff2

    # Test state dist
    for i in range(5):
        assert set(critic.state_dist(i, coeff).support()) == {i}
