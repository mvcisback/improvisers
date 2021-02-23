import pytest

import random

import improvisers as RCI
from improvisers.tabular import Dist


def test_mdp_policy_smoke():
    game_graph = RCI.ExplicitGameGraph(
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
    actor = RCI.solve(game_graph, psat=0.8)
    policy = actor.improvise()

    random.seed(1)
    observation = None
    while True:
        try:
            action, state_dist = policy.send(observation)
            observation = state_dist.sample(), state_dist  # Env player move.
        except StopIteration as e:
            assert game_graph.label(observation[0]) is e.value
            break


def test_stochastic_game_policy_smoke():
    game_graph = RCI.ExplicitGameGraph(
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

    with pytest.raises(ValueError):   # Unachievable psat, achievable entropy.
        RCI.solve(game_graph, percent_entropy=0, psat=1/3 + 0.1)

    with pytest.raises(ValueError):   # Achievable psat, unachievable entropy.
        RCI.solve(game_graph, percent_entropy=0.99, psat=1/3)

    actor = RCI.solve(game_graph, psat=1/3)  # Max ent
    policy = actor.improvise()
    observation = None
    while True:
        try:
            action, state_dist = policy.send(observation)
            if action == 4:
                observation = 3, Dist({2: 0.8, 3: 0.2})
            else:
                observation = state_dist.sample(), state_dist  # Env move.

        except StopIteration as e:
            assert game_graph.label(observation[0]) is e.value
            break
