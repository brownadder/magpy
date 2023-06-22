import pytest

from magpy import Sigma


@pytest.mark.parametrize("x, y, z, expected_spins", 
    [
        ({}, {}, {}, {}),
        (1, {}, {}, {1 : 'x'}),
        (1, 2, 3, {1 : 'x', 2 : 'y', 3 : 'z'}),
        (3, {}, 1, {1 : 'z', 3 : 'x'}),
        ({1,2}, 3, {}, {1 : 'x', 2 : 'x', 3 : 'y'})
    ])
def test_instantiation(x, y, z, expected_spins):
    assert Sigma(x, y, z).spins == expected_spins


@pytest.mark.parametrize("instance, expected_spins",
    [
        (Sigma.X(), {1 : 'x'}),
        (Sigma.Y(), {1 : 'y'}),
        (Sigma.Z(), {1 : 'z'}),
        (Sigma.Y(2), {2 : 'y'}),
        (Sigma.Z({1,2}), {1 : 'z', 2 : 'z'})
    ])
def test_xyz_instantiation(instance, expected_spins):
    assert instance.spins == expected_spins


@pytest.mark.parametrize("s1, s2, expected_spins",
    [
        (Sigma(), Sigma(), {}),
        (Sigma(x=1), Sigma(), {1 : 'x'}),
        (Sigma(x=1), Sigma(y=2), {1 : 'x', 2 : 'y'}),
        (Sigma(z=1), Sigma(x=1), {1 : 'zx'}),
        (Sigma(x=1), Sigma(z=1), {1 : 'xz'}),
        (Sigma(scale=0), Sigma(x=1), {})
    ])
def test_multiplication(s1, s2, expected_spins):
    assert (s1 * s2).spins == expected_spins


@pytest.mark.parametrize("scale, s1, expected_scale",
    [
        (1, Sigma(), 1),
        (2, Sigma(x=1), 2),
        (3j + 4, Sigma(x=2), 3j + 4),
        (0, Sigma(x=1), 0)
    ])
def test_scalar_multiplication(scale, s1, expected_scale):
    assert (scale * s1).scale == expected_scale


def test_negation():
    assert (-Sigma()).scale == -1