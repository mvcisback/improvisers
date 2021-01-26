import improvisers as RCI


def test_game_graph_smoke():
    game_graph = RCI.ExplicitGameGraph(
        root=1,
        graph={
            1: ('p1', {2, 3}),
            2: ('p1', {5, 4}),
            3: ('env', {2: 0.3, 6: 0.7}),
            5: ('env', {4: 0.5, 6: 0.5}),
            4: (True, {}),
            6: (False, {}),
        }
    )

    assert set(game_graph.nodes()) == {1, 2, 3, 4, 5, 6}

    assert game_graph.label(1) == 'p1'
    assert game_graph.label(2) == 'p1'
    assert game_graph.label(3) == 'env'
    assert game_graph.label(5) == 'env'
    assert game_graph.label(4) is True
    assert game_graph.label(6) is False

    assert game_graph.root == 1

    for node in game_graph.nodes():
        for action in game_graph.actions(node):
            assert action.size == 1
