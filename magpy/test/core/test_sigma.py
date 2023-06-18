import pytest

from magpy import Sigma

@pytest.mark.parametrize("x, y, z, expected_spins", 
    [
        (None, None, None, {}),
        (1, None, None, {1 : 'x'}),
        (1, 2, 3, {1 : 'x', 2 : 'y', 3 : 'z'}),
        (3, None, 1, {1 : 'z', 3 : 'x'}),
        ({1,2}, 3, None, {1 : 'x', 2 : 'x', 3 : 'y'})
    ])

def test_instantiation(x, y, z, expected_spins):
    assert Sigma(x, y, z).spins == expected_spins

@pytest.mark.parametrize("s1, s2, expected_spins",
    [
        (Sigma(), Sigma(), {}),
        (Sigma(x=1), Sigma(), {1 : 'x'}),
        (Sigma(x=1), Sigma(y=2), {1 : 'x', 2 : 'y'}),
        (Sigma(z=1), Sigma(x=1), {1 : 'zx'}),
        (Sigma(x=1), Sigma(z=1), {1 : 'xz'})
    ])

def test_multiplication(s1, s2, expected_spins):
    assert (s1 * s2).spins == expected_spins