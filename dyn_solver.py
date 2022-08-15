import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal

def rayleigh_damping(M, K, alpha, beta):
    assert len(M) == len(K)
    C = alpha*M + beta*K
    return C

M = np.array([[5, 0], [0, 2]]) # Change these arrays to achieve different systems. Should scale to any DOF but i havent checked that yet.
K = np.array([[25, -3], [-3, 20], [0, 3]]) #

alpha = -1.2 # Controls the linear combination that forms the damping matrix.
beta = 0.3

C = rayleigh_damping(M, K, alpha, beta)

class AbsorberSystem:
    '''
    Computes required specifications to dampen system natural frequecies.
    '''
    def __init__(self):
        pass

class BasicSystem:
    '''
    Creates and solves a MDOF system given M, K and, optionally, C.

    TODO: Add inital conditions for system response.
    '''
    def __init__(self, M: np.array, K: np.array, C = None):
        self.M = M
        self.C = C
        self.K = K

        self.eigenvalues = None
        self.eigenvectors = None
        self.wn = None
        self.zeta = None
        self.wd = None
        self.modes = None
        self.lti = None

        self.n = len(M)

        assert len(M) == len(K)

        self.compute_sys()
        print(f'{self.n} dimensional system solved!\n')
        self.display()

    def display(self):
        if self.C is None:
            print(f'Eigenvalues: {self.eigenvalues}\n')
            print(f'Eigenvectors: {self.eigenvectors}\n')
            print(f'Natural frequencies: {self.wn}\n')
            print(f'Modes: {self.modes}\n')
        else:
            print(f'Eigenvalues: {self.eigenvalues}\n')
            print(f'Eigenvectors: {self.eigenvectors}\n')
            print(f'Natural frequencies: {self.wn}\n')
            print(f'Damping ratios: {self.zeta}\n')
            print(f'Damped natural frequencies: {self.wd}\n')
            print(f'Modes: {self.modes}\n')

    def set_M(self, M: np.array):
        self.M = M
        self.compute_sys()

    def set_C(self, C: np.array):
        self.C = C
        self.compute_sys()

    def set_K(self, K: np.array):
        self.K = K
        self.compute_sys()
    
    def compute_sys(self):
        if self.C is None:
            L = la.cholesky(self.M)
            self.eigenvalues, self.eigenvectors = la.eig(np.transpose(la.solve(L, np.transpose(la.solve(L, self.K)))))
            self.wn = np.sqrt(self.eigenvalues)
        else:
            self.eigenvalues, self.eigenvectors = self.compute_eigen()
            self.wn = np.absolute(self.eigenvalues)[:self.n]
            self.wd = np.imag(self.eigenvalues)[:self.n]
            self.zeta = (-np.real(self.eigenvalues)/np.absolute(self.eigenvalues))[:self.n]
            self.lti = self.linear_time_system()
        self.modes = self.solve_modes()

    def state_space(self):
        zero = np.zeros((self.n, self.n))
        identity = np.identity(self.n)

        A = np.block([[zero, identity], [la.solve(-self.M, self.K), la.solve(-self.M, self.C)]])

        return A

    @staticmethod
    def sort_eigenvalues(evals):
        evals_float = np.around(evals, decimals=10)
        a = np.imag(evals_float)
        b = np.absolute(evals_float)
        ind = np.lexsort((b, a))

        pos = [i for i in ind[len(a) // 2:]]
        neg = [i for i in ind[:len(a) // 2]]

        idx = np.array([pos, neg]).flatten()
        return idx

    def compute_eigen(self):
        A = self.state_space()
        eigenvalues, eigenvectors = la.eig(A)

        idx = self.sort_eigenvalues(eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]

    def solve_modes(self):
        if self.C is None:
            L = la.cholesky(self.M)
            return la.solve(L, self.eigenvectors)
        else:

            # Still figuring out how to get the modes from the damped system.
            return self.eigenvectors

    def absorption(self):
        # TODO: Add way to calculate absorber mass and spring stiffness automagically.
        pass

    def linear_time_system(self):
        zeros = np.zeros((self.n, self.n))
        identity = np.identity(self.n)

        B2 = identity
        A = self.state_space()
        B = np.vstack([zeros, la.solve(self.M, B2)])

        Cd = identity
        Cv = zeros
        Ca = zeros

        C = np.hstack((Cd - Ca @ la.solve(self.M, self.K),
                       Cv - Ca @ la.solve(self.M, self.C)))

        D = Ca @ la.solve(self.M, B2)

        return signal.lti(A, B, C, D)

    def time_response(self, F, t, ic = None):
        return signal.lsim(self.lti, F, t, X0=ic)

    def frequency_response(self, F = None, w = None, modes = None):
        rows = self.lti.inputs
        cols = self.lti.outputs

        B = self.lti.B
        C = self.lti.C
        D = self.lti.D

        evals = self.eigenvalues
        evecs = self.eigenvectors
        evecs_inv = la.inv(evecs)

        if w is None:
            w = np.linspace(0, np.max(evals.imag)*1.5, 1000)

        if modes is not None:
            n = self.n
            m = len(modes)

            idx = np.zeros((2 * m), int)
            idx[0:m] = modes
            idx[m:] = range(2 * n)[-m:]

            evals = evals[np.ix_(idx)]
            evecs = evecs[np.ix_(range(2 * n), idx)]
            evecs_inv = evecs_inv[np.ix_(idx, range(2 * n))]

        mag = np.empty((cols, rows, len(w)))
        phase = np.empty((cols, rows, len(w)))

        for wi, w1 in enumerate(w):
            diag = np.diag([1 / (1j * w1 - lam) for lam in evals])
            if F is None:
                H = C @ evecs @ diag @ evecs_inv @ B + D
            else:
                H = (C @ evecs @ diag @ evecs_inv @ B + D) @ F[wi]

            magh = abs(H)
            angh = np.rad2deg((np.angle(H)))

            mag[:, :, wi] = magh
            phase[:, :, wi] = angh

        return w, mag, phase

    def plot_frequency_response(self, out, inp, modes=None, ax0=None, ax1=None, **kwargs):
        if ax0 is None or ax1 is None:
            fig, ax = plt.subplots(2)
            if ax0 is not None:
                _, ax1 = ax
            if ax1 is not None:
                ax0, _ = ax
            else:
                ax0, ax1 = ax
        omega, magdb, phase = self.frequency_response(modes=modes)

        ax0.plot(omega, magdb[out, inp, :], **kwargs)
        ax1.plot(omega, phase[out, inp, :], **kwargs)
        for ax in [ax0, ax1]:
            ax.set_xlim(0, max(omega))
            ax.yaxis.set_major_locator(
                mpl.ticker.MaxNLocator(prune='lower'))
            ax.yaxis.set_major_locator(
                mpl.ticker.MaxNLocator(prune='upper'))

        ax0.text(.9, .9, 'Output %s' % out,
                 horizontalalignment='center',
                 transform=ax0.transAxes)
        ax0.text(.9, .7, 'Input %s' % inp,
                 horizontalalignment='center',
                 transform=ax0.transAxes)

        ax0.set_ylabel('Magnitude')
        ax1.set_ylabel('Phase')
        ax1.set_xlabel('Frequency (rad/s)')

        return ax0, ax1

    def plot_time_response(self, F, t, ic=None, out=None, ax=None):
        if ax is None:
            fig, axs = plt.subplots(self.lti.outputs, 1, sharex=True)

            fig.suptitle('Time response', fontsize=12)
            plt.subplots_adjust(hspace=0.01)

        if out is not None:
            raise NotImplementedError('Not implemented yet for specific outputs.')

        t, yout, xout = self.time_response(F, t, ic=ic)

        for i, ax in enumerate(axs):
            ax.plot(t, yout[:, i])

        # set the same y limits
        min_ = min([ax.get_ylim()[0] for ax in axs])
        max_ = max([ax.get_ylim()[1] for ax in axs])
        lim = max(abs(min_), max_)

        for i, ax in enumerate(axs):
            ax.set_ylim([-lim, lim])
            ax.set_xlim(t[0], t[-1])
            ax.set_ylabel('Amp. output %s (m)' % i, fontsize=8)

        axs[-1].set_xlabel('Time (s)')

        return axs

sys = BasicSystem(M, K, C)

