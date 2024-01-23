import pytest
import numpy as np

from magpy import PauliString as PS, FunctionProduct as FP, HamiltonianOperator as HO


@pytest.mark.parametrize("x, y, z, expected_qubits",
    [
        (None, None, None, {}),
        (1, {}, {}, {1: 'x'}),
        (1, 2, {3, 4}, {1: 'x', 2: 'y', 3: 'z', 4: 'z'})
    ])
def test_instantiation(x, y, z, expected_qubits):
    assert PS(x, y, z).qubits == expected_qubits


@pytest.mark.parametrize("instance, expected_qubits",
    [
        (PS.X(), {1: 'x'}),
        (PS.X(2), {2: 'x'}),
        (PS.X(1, 2), {1: 'x', 2: 'x'})
    ])
def test_xyz_instantiation(instance, expected_qubits):
    assert instance.qubits == expected_qubits


@pytest.mark.parametrize("p1, p2, expected_product",
    [
        (PS(), PS.X(), PS.X()),
        (PS.X(), PS.X(), PS()),
        (PS.X(), PS.Y(), 1j*PS.Z())
    ])
def test_multiplication(p1, p2, expected_product):
    assert p1*p2 == expected_product


def test_scalar_multiplication():
    assert ((3j + 4) * PS.X()).scale == 3j + 4


def test_negation():
    assert (-PS.X()).scale == -1


@pytest.mark.parametrize("f, p, expected_product",
    [
        (np.sin, PS.X(), HO([np.sin, PS.X()])),
        (FP(np.sin), PS.X(), HO([FP(np.sin), PS.X()]))
    ])
def test_function_multiplication(f, p, expected_product):
    assert f*p == expected_product
