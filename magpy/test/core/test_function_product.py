import pytest
import numpy as np

from magpy import FunctionProduct as FP, PauliString as PS


@pytest.mark.parametrize("args, expected_funcs",
    [
        ([], {}),
        ([1], {}),
        ([np.sin], {np.sin: 1}),
        ([np.sin, np.cos, np.cos], {np.sin: 1, np.cos: 2})
    ])
def test_instantiation(args, expected_funcs):
    assert FP(*args).funcs == expected_funcs


def test_scalar_multiplication():
    assert (2 * FP()).scale == 2


@pytest.mark.parametrize("f, g, expected_funcs",
    [
        (FP(np.sin), FP(), {np.sin: 1}),
        (FP(np.sin), FP(np.sin), {np.sin: 2}),
        (FP(np.sin), FP(np.cos), {np.sin: 1, np.cos: 1})
    ]
)
def test_multiplication(f, g, expected_funcs):
    assert (f * g).funcs == expected_funcs


def test_negation():
    assert (-FP().scale) == -1


def test_pauli_string_multiplication():
    assert (FP(np.sin) * PS.X()).data == {FP(np.sin): PS.X()}
