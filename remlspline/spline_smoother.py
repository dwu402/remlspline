import numpy as np
from scipy.optimize import minimize
from numbers import Number

class SplineSmoother():
    def __init__(self, t, y, k=3, s=None, nk=None):
        """ Spline Smoother: 1D data

        Arguments
        ----------
        t: independent variable
        y: dependent variable
        k: spline degree
        s: smoothing coefficient (> 0)
        nk: number of knots (defaults to 0.1 of length of input)
        """
        self.k = k
        self.t = t
        self.y = y
        self.d = 0

        if nk is not None:
            self.K = nk
        else:
            self.K = int(len(t)//10)
        self.knots = None
        self.build_knots(t)
        
        self.n = len(t)
        self.m = 2

        self.X = self.build_X(t)

        self.Z = self.build_Z(t)

        if s is None:
            self.autosmooth(y)
        else:
            self.s = s

        self.b = None
        self.u = None
        self.compute_coefs()

    def __call__(self, ti):
        if isinstance(ti, Number):
            ti = [ti]
        return self.build_X(ti)@self.b + self.build_Z(ti)@self.u

    def build_knots(self, t):
        """Constructs uniformly spaced knots"""
        lims = np.min(t), np.max(t)
        self.knots = np.linspace(*lims, self.K)

    def build_X(self, t):
        """Constructs the X matrix (fixed effects, linear trend)"""
        return np.hstack([np.ones((len(t), 1)), np.array(t).reshape((-1, 1))])

    def build_Z(self, t, d=None):
        """Constructs the Z matrix (random effects, nonlinear spline portion)"""
        bas = np.array([[ti-k if ti >= k else 0 for k in self.knots] for ti in t])
        if d is None:
            return np.hstack([bas**i for i in range(1, self.k+1)])
        else:
            bas_zero = (bas > 0).astype(float)
            return np.hstack([np.prod(np.arange(i-d+1, i+1)) * (bas**(i-d) if i>d else bas_zero) for i in range(1, self.k+1)])

    def spl_coefs(self, l, y):
        """Computes the spline coefficients given lambda l and data y"""
        X, Z, K = self.X, self.Z, self.K*self.k
        A = np.vstack([np.hstack([X.T@X, X.T@Z]), np.hstack([Z.T@X, Z.T@Z+l**2*np.eye(K)])])
        return np.linalg.solve(A, np.vstack([X.T, Z.T])@y)

    def set_s(self, s):
        self.s = s
        self.compute_coefs()

    def compute_coefs(self):
        b1, b2, *u = self.spl_coefs(self.s, self.y)
        self.b = np.array([b1, b2]).T
        self.u = u

    def build_likelihood(self, y):
        """Builds the REML lieklihood for the spline as a function of smoothing bandwidth lambda
        
        Assumes y = Xb + Zu + e
        """
        dof = self.n - self.m
        n, m, X, Z = self.n, self.m, self.X, self.Z
        def likelihood(p):
            lm, = p
            Sig = 1/lm**2 * Z@Z.T + np.eye(n)
            r = y - X@self.spl_coefs(lm, y)[:m]
            slm = 1/dof * r.T@np.linalg.solve(Sig, r)
            lk = (dof
                  + np.log(np.linalg.det(Sig))
                  + dof*np.log(slm*dof)
                  + np.log(np.linalg.det(X.T@np.linalg.solve(Sig, X))))
            return 0.5*lk
        return likelihood

    def autosmooth(self, y):
        """Optimise for best smoothing bandwidth (lambda)"""
        likelihood = self.build_likelihood(y)
        res = minimize(likelihood, [1], bounds=[(1e-6, None)])
        self.s = res.x[0]

    def derivative(self, d=1):
        """Returns a function that evaluates the spline derivative"""
        def deriv_spl(ti):
            """The d-th derivative of the spline"""
            if isinstance(ti, Number):
                ti = [ti]
            if d == 1:
                X = np.hstack([np.zeros((len(ti), 1)), np.ones((len(ti), 1))])
            else:
                X = np.zeros((len(ti), 2))
            Z = self.build_Z(ti, d=d)
            return X@self.b + Z@self.u
        return deriv_spl
