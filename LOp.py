from magpy import *
import magpy

class LOp:
    def __init__(self, n, pos, ham):
        array = [np.eye(2) for _ in range(n)]
        array[pos - 1] = ham

        self.data = kron(array)

    def __call__(self):
        return self.data

    def isHermitian(self):
        return np.array_equal(self.data.conj().T, self.data)


def evaluate(H, t):
    # evaluate H at t
    out = 0

    for coeff in H:
        if isinstance(H[coeff], LOp):
            H[coeff] = [H[coeff]]

        # coefficient is constant
        if isinstance(coeff, (int, float)):
            for matrix in H[coeff]:
                out = out +  matrix()*coeff

        # coefficient is a function
        else:
            for matrix in H[coeff]:
                out = out + matrix()*coeff(t)

    return out


def _mag1(H, t0, tf):
    # first term of Magnus expansion
    total = 0

    for coeff in H:
        if isinstance(coeff, (int, float)):
            for matrix in H[coeff]:
                total = total + matrix()*coeff*(tf - t0)
        else:
            c = scipy.integrate.quad(coeff, t0, tf)[0]
            for matrix in H[coeff]:
                total = total + c*matrix()

    return liouvillian(total)

def _mag2(H, t0, tf):
    return 0 # TODO


def setup(H):
    # change instances of single matrices in H to lists with single items
    for coeff in H:
        if isinstance(H[coeff], LOp):
            H[coeff] = [H[coeff]]


def f(t): return t
def g(t): return t**2
def h(t): return 0
omega = 1

H_new = {f : [LOp(2,1,sigmax()), LOp(2,2,sigmax())], g : LOp(2,2,sigmay())}
H_old = [[f, h, 0], [f, g, 0]]

setup(H_new)
print(_mag1(H_new, 0, 1) - magpy._magnus_first_term(H_old, np.zeros((4, 4)), 0, 1))