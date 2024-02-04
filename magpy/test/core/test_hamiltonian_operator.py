import pytest
import numpy as np

from magpy import PauliString as PS, HamiltonianOperator as HO


@pytest.mark.parametrize("pairs, expected_data",
    [
        ([], {}),
        ([[np.sin, PS.X()]], {np.sin: PS.X()}),
        ([[np.sin, PS.X()], [np.sin, PS.Y()]], {np.sin: [PS.X(), PS.Y()]}),
        ([[2, PS.X()]], {1: 2*PS.X()})
    ])
def test_instantiation(pairs, expected_data):
    assert HO(*pairs).data == expected_data


def test_scalar_multiplication():
    assert (2 * HO([np.sin, PS.X()])).data[np.sin].scale == 2


def test_negation():
    assert (-HO([np.sin, PS.X()])).data[np.sin].scale == -1


def test_pauli_string_multiplication():
    assert (PS.X() * HO([np.sin, PS.Y()])).data[np.sin] == -1j*PS.Z()
