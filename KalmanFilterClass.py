import numpy as np

try:
    from scipy.linalg import expm
except ImportError:
    expm = None


class KalmanFilter:
    """
    Analytical linear Kalman filter matched to PlantClass.py.
    """

    def __init__(self, system, dt, process_noise=None, observation_noise=None):
        self.system = system
        self.dt = float(dt)

        self.n = system.x_k
        self.m = system.y_k

        self.A = np.array(system.A_lin, dtype=float, copy=True)
        self.C = np.array(system.C, dtype=float, copy=True)

        self.Phi = self._make_phi(self.dt)
        self.Q_d = self.dt * self._make_covariance(process_noise, self.n, system.V_n)
        self.R_d = self.dt * self._make_covariance(observation_noise, self.m, system.V_d)

        self.reset()

    def _make_covariance(self, noise_value, dim, fallback):
        if noise_value is None:
            return np.array(fallback, dtype=float, copy=True)

        noise_value = np.asarray(noise_value, dtype=float)
        if noise_value.ndim == 0:
            return float(noise_value) * np.eye(dim)
        if noise_value.ndim == 1:
            if noise_value.size != dim:
                raise ValueError(f"Noise guess must have length {dim}, got {noise_value.size}")
            return np.diag(noise_value)
        if noise_value.shape != (dim, dim):
            raise ValueError(f"Noise guess must have shape {(dim, dim)}, got {noise_value.shape}")
        return noise_value.copy()

    def _make_phi(self, dt):
        if expm is not None:
            return expm(self.A * dt)
        return np.eye(self.n) + dt * self.A

    def reset(self):
        self.x = np.zeros(self.n)
        self.x_prior = np.zeros(self.n)

        self.P = np.eye(self.n)
        self.P_prior = np.eye(self.n)

        self.K = np.zeros((self.n, self.m))
        self.innovation = np.zeros(self.m)
        self.S = np.eye(self.m)

        self.initialised = False

    def setup(self, x0=None, P0=None):
        if x0 is None:
            x0 = np.array(self.system.x0_lin, dtype=float, copy=True)
        else:
            x0 = np.array(x0, dtype=float, copy=True)

        if P0 is None:
            P0 = np.eye(self.n)
        else:
            P0 = np.array(P0, dtype=float, copy=True)

        self.x = x0.copy()
        self.x_prior = x0.copy()
        self.P = P0.copy()
        self.P_prior = P0.copy()
        self.initialised = True

    def predict(self):
        self.x_prior = self.Phi @ self.x
        self.P_prior = self.Phi @ self.P @ self.Phi.T + self.Q_d
        return self.x_prior.copy()

    def update(self, y):
        y = np.array(y, dtype=float, copy=True)

        if not self.initialised:
            self.setup()

        self.predict()

        self.innovation = y - self.C @ self.x_prior
        self.S = self.C @ self.P_prior @ self.C.T + self.R_d
        self.K = self.P_prior @ self.C.T @ np.linalg.inv(self.S)

        self.x = self.x_prior + self.K @ self.innovation

        I = np.eye(self.n)
        KC = self.K @ self.C
        self.P = (I - KC) @ self.P_prior @ (I - KC).T + self.K @ self.R_d @ self.K.T
        return self.x.copy()
