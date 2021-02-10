import improvisers as RCI


def test_game_graph_explicit_smoke():
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

    assert set(RCI.dfs_nodes(game_graph)) == {1, 2, 3, 4, 5, 6}

    assert game_graph.label(1) == 'p1'
    assert game_graph.label(2) == 'p1'
    assert isinstance(game_graph.label(3), RCI.ExplicitDist)
    assert game_graph.label(3).data == {2: 0.3, 6: 0.7}
    assert isinstance(game_graph.label(5), RCI.ExplicitDist)
    assert game_graph.label(5).data == {4: 0.5, 6: 0.5}
    assert game_graph.label(4) is True
    assert game_graph.label(6) is False

    assert game_graph.root == 1
