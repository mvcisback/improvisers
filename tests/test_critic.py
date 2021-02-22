# flake8: noqa

from pytest import approx

import math

import improvisers as I


def test_stochastic_game_critic():
    game_graph = I.ExplicitGameGraph(
        root=5,
        graph={
            0: (False, {}),
            1: (True, {}),
            2: ('env', {0: 2/3, 1: 1/3}),
            3: ('p1', {0, 2}),
            4: ('p2', {2, 3}),
            5: ('p1', {4, 3}),
        }
    )
    critic = I.TabularCritic.from_game_graph(game_graph)

    coeff = math.log(8)

    assert critic.value(0, coeff) == 0
    assert critic.value(1, coeff) == coeff
    assert critic.value(2, coeff) == approx(math.log(2))
    assert critic.value(3, coeff) == approx(math.log(3))
    assert critic.value(4, coeff) == approx(math.log(2))
    assert critic.value(5, coeff) == approx(math.log(5))

    assert critic.entropy(0, coeff) == 0
    assert critic.entropy(1, coeff) == 0
    assert critic.entropy(2, coeff) == 0
    assert critic.entropy(3, coeff) == approx(
        2/3*(math.log(3/2) + critic.entropy(2, coeff)) + 
        1/3*(math.log(3)   + 0                       )
    )
    assert critic.entropy(4, coeff) == critic.entropy(2, coeff)
    assert critic.entropy(5, coeff) == approx(
        3/5*(math.log(5/3) + critic.entropy(3, coeff)) +
        2/5*(math.log(5/2) + critic.entropy(2, coeff))
    )

    # psat is monotonic
    for i in range(6):
        coeff = 10**(1 - i)
        psat = critic.psat(game_graph.root, coeff)
        assert 0 <= psat < 1
    assert critic.psat(game_graph.root, float('inf')) == 1/3



def test_stochastic_game_critic_pareto():
    game_graph = I.ExplicitGameGraph(
        root=5,
        graph={
            0: (False, {}),
            1: (True, {}),
            2: ('env', {0: 2/3, 1: 1/3}),
            3: ('p1', {0, 2}),
            4: ('p2', {2, 3}),
            5: ('p1', {4, 3}),
        }
    )
    critic = I.ParetoCritic.from_game_graph(game_graph, tol=1e-2)

    coeff = math.log(8)

    assert critic.value(0, coeff) == 0
    assert critic.value(1, coeff) == coeff
    assert critic.value(2, coeff) == approx(math.log(2))
    assert critic.value(3, coeff) == approx(math.log(3))
    assert critic.value(4, coeff) == approx(math.log(2))
    assert critic.value(5, coeff) == approx(math.log(5))

    assert critic.entropy(0, coeff) == 0
    assert critic.entropy(1, coeff) == 0
    assert critic.entropy(2, coeff) == 0
    assert critic.entropy(3, coeff) == approx(
        2/3*(math.log(3/2) + critic.entropy(2, coeff)) + 
        1/3*(math.log(3)   + 0                       )
    )
    assert critic.entropy(4, coeff) == critic.entropy(2, coeff)
    assert critic.entropy(5, coeff) == approx(
        3/5*(math.log(5/3) + critic.entropy(3, coeff)) +
        2/5*(math.log(5/2) + critic.entropy(2, coeff))
    )

    # psat is monotonic
    for i in range(6):
        coeff = 10**(1 - i)
        psat = critic.psat(game_graph.root, coeff)
        assert 0 <= psat < 1
