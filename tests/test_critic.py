from pytest import approx

import math

import improvisers as I


def test_deterministic_critic():
    game_graph = I.ExplicitGameGraph(
        root=4,
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
    coeff2 = critic.match_psat(4, psat)
    psat2 = critic.psat(4, coeff2)

    assert approx(psat) == psat2
    assert approx(coeff) == coeff2

    # Test state dist
    for i in range(5):
        assert set(critic.state_dist(i, coeff).support()) == {i}

    # psat is monotonic
    for i in range(6):
        coeff = 10**(1 - i)
        psat = critic.psat(game_graph.root, coeff)
        assert 0 <= psat < 1
    assert critic.psat(game_graph.root, float('inf')) == 1


def test_mdp_critic():
    game_graph = I.ExplicitGameGraph(
        root=7,
        graph={
            0: (False, {}),
            1: (True, {}),
            2: ('env', {0: 1/3, 1: 2/3}),
            3: ('p1', {1, 2}),
            4: ('env', {0: 1/3, 3: 2/3}),
            5: ('p1', {3, 4}),
            6: ('env', {0: 1/3, 5: 2/3}),
            7: ('p1', {5, 6}),
        }
    )
    critic = I.TabularCritic.from_game_graph(game_graph)

    def env_op(val, p=2/3):
        return p * val

    def p1_op(val, p=2/3):
        return math.log(math.exp(val) + math.exp(val)**p)

    # Test Values
    coeff = math.log(2)
    assert critic.value(0, coeff) == 0
    assert critic.value(1, coeff) == coeff
    for i in range(2, 8):
        val = critic.value((i & -2) - 1, coeff)
        expected = p1_op(val) if i % 2 else env_op(val)
        assert critic.value(i, coeff)== approx(expected)

    def p1_prob_hi(i):
        val = critic.value(i, coeff) 
        return (1 + math.exp(val*(2/3 - 1)))**-1

    # Test action probabilities of policy.
    for i in [3, 5, 7]:
        action = i - 2
        expected = p1_prob_hi(action)
        assert critic.action_dist(i, coeff).prob(action) == approx(expected)

    assert critic.entropy(0, coeff) == 0
    assert critic.entropy(1, coeff) == 0
    for i in range(2 + 1, 8, 2):  # P1 node entropies.
        prob = p1_prob_hi(i - 2)

        expected = (    prob) * (-math.log(prob)     + critic.entropy(i - 2, coeff)) \
                 + (1 - prob) * (-math.log(1 - prob) + critic.entropy(i - 1, coeff))

        assert critic.entropy(i, coeff) == approx(expected)

    for i in range(2, 8, 2):     # Env node entropies.
        prob = 2 / 3

        expected = (    prob) * (-math.log(prob)     + critic.entropy(i - 1, coeff)) \
                 + (1 - prob) * (-math.log(1 - prob) + 0                           )

        assert critic.entropy(i, coeff) == approx(expected)

    # Test psat and rationality are approximate inverses.
    psat = critic.psat(4, coeff)
    coeff2 = critic.match_psat(4, psat)
    psat2 = critic.psat(4, coeff2)

    assert approx(psat) == psat2
    assert approx(coeff) == coeff2

    # Test state dist
    for i in range(1, 8):
        if i & 1:
            assert set(critic.state_dist(i, coeff).support()) == {i}
        else:
            assert set(critic.state_dist(i, coeff).support()) == {0, i - 1}
            assert critic.state_dist(i, coeff).prob(i - 1) == approx(2/3)

    # psat is monotonic
    for i in range(6):
        coeff = 10**(1 - i)
        psat = critic.psat(game_graph.root, coeff)
        assert 0 <= psat < 1
    assert critic.psat(game_graph.root, float('inf')) == 1
