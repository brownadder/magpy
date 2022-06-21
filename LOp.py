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


def setup(H): # TODO: implement 2nd term Magnus
    # pre-calculates liouvillians
    H_new = {}

    for coeff in H:
        # change instances of single matrices in H to arrays
        if isinstance(H[coeff], LOp):
            H[coeff] = [H[coeff]]

        # calculate liouvillian of each matrix
        matrices = []
        for matrix in H[coeff]:
            matrices.append(liouvillian(matrix()))

        H_new[coeff] = matrices

    return H_new


def _mag1(H, t0, tf):
    # uses pre-calculated liouvillians of matrices in H
    total = 0

    for coeff in H:
        if isinstance(coeff, (int, float)):
            for matrix in H[coeff]:
                total = total + matrix*coeff*(tf - t0)
        else:
            c = scipy.integrate.quad(coeff, t0, tf)[0]
            for matrix in H[coeff]:
                total = total + c*matrix

    return total


def _mag2(H, t0, tf): # TODO
    # second term of Magnus expansion
    # uses pre-calculated liouvillians of commutators of matrices
    total = 0

    return total


def lvnsolve_new(H, rho0, tlist):
    states = [vec(rho0)]
    for i in range(len(tlist) - 1):
        omega = (_mag1(H, tlist[i], tlist[i+1])
                 + _mag2(H, tlist[i], tlist[i+1]))

        states.append(scipy.linalg.expm(omega) @ states[i])
        states[i] = unvec(states[-1])

    return states


# testing
def f(t): return t
def g(t): return t**2
def h(t): return 0
omega = 1

H_old = [[f, h, 0], [f, g, 0]]

# dict = {(x,y):np.zeros((2,2)) for x in [1,2,3] for y in [1,2,3] if x!=y}

H_new = {f : [LOp(2,1,sigmax()), LOp(2,2,sigmax())], g : LOp(2,2,sigmay())}