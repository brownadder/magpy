# Contributing

## Coding standards

For the most part, MagPy follows [NumPy's style guide](https://numpydoc.readthedocs.io/en/latest/format.html) for Python code and documentation.

The core points are as follows:

- PEP 8 (use [flake8](https://pypi.org/project/flake8/) and [pylint](https://pypi.org/project/pylint/) for code checking)
- NumPy-style docstrings
- 120 character line limit
- 80 character docstring line limit

## Setting up a development environment

We strongly recommend working in a [virtual environment](https://docs.python.org/3/library/venv.html).

Install the [development requirements](requirements/dev.txt) via [pip](https://pip.pypa.io/en/stable/installation/); or, for VS Code, use the extensions for [flake8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8) and [pylint](https://marketplace.visualstudio.com/items?itemName=ms-python.pylint). See [here](./.vscode/settings.json) for workspace settings.

## Unit tests

MagPy uses the [pytest](https://docs.pytest.org/en/7.4.x/contents.html) framework for writing unit tests.

Tests go in [magpy/test/](magpy/test/), which mirrors the structure of [magpy/](magpy), prepending each file name with `test_`.