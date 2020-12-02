import improvisers as I


def test_game_graph_smoke():
    game_graph = I.ExplicitGameGraph(
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

