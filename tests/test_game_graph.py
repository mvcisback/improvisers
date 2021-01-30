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

    assert set(game_graph.nodes()) == {1, 2, 3, 4, 5, 6}

    assert game_graph.label(1) == 'p1'
    assert game_graph.label(2) == 'p1'
    assert isinstance(game_graph.label(3), RCI.ExplicitDist)
    assert game_graph.label(3).data == {2: 0.3, 6: 0.7}
    assert isinstance(game_graph.label(5), RCI.ExplicitDist)
    assert game_graph.label(5).data == {4: 0.5, 6: 0.5}
    assert game_graph.label(4) is True
    assert game_graph.label(6) is False

    assert game_graph.root == 1


def test_game_graph_implicit_smoke():
    game_graph = RCI.ImplicitGameGraph(
        root=False,
        player=lambda _: 'p1',
        accepting=lambda i: i,
        transition=lambda n, c: bool(n ^ c),
        actions=lambda _: [False, True],
        horizon=2,
    )

    assert set(game_graph.nodes()) == {
        (0, False),
        (1, False),
        (1, True),
        (2, False),
        (2, True),
    }

    assert game_graph.label((0, False)) == 'p1'
    assert game_graph.label((1, False)) == 'p1'
    assert game_graph.label((1, True)) == 'p1'
    assert game_graph.label((2, False)) is False
    assert game_graph.label((2, True)) is True
