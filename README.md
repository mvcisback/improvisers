# Improvisers

Library for synthesizing Entropic Reactive Control Improvisers for
stochastic games.

[![Build Status](https://cloud.drone.io/api/badges/mvcisback/improvisers/status.svg)](https://cloud.drone.io/mvcisback/improvisers)
[![Docs](https://img.shields.io/badge/API-link-color)](https://mvcisback.github.io/improvisers)
[![codecov](https://codecov.io/gh/mvcisback/improvisers/branch/master/graph/badge.svg)](https://codecov.io/gh/mvcisback/improvisers)
[![PyPI version](https://badge.fury.io/py/improvisers.svg)](https://badge.fury.io/py/improvisers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# About


# Installation

If you just need to use `improvisers`, you can just run:

`$ pip install improvisers`

For developers, note that this project uses the
[poetry](https://poetry.eustace.io/) python package/dependency
management tool. Please familarize yourself with it and then
run:

`$ poetry install`

# Usage

```python
from improvisers import policy, ExplicitGameGraph

game_graph = ExplicitGameGraph(
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

actor = solve(game_graph, psat=1/3)
policy = actor.improvise()
```
