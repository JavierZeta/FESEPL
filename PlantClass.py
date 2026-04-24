import numpy as np

class Plant:
    def __init__(self, system='SMD', N=5, v_n = 0.0001, v_d = 0.0001, seed=None):
        """
        Parameters:
          dt: time step.
          Time: total simulation time.
          system: system type ('2D_masses';
                               'SMD';
                               'coupledSMD';
                               'cartpole').
        """

        self.system = system
        self.rng = np.random.default_rng(seed)
        
        if system == '2D_masses':
            self.setup_Masses_2D(N)

        elif system == 'SMD':
            self.setup_SMD()

        elif system == 'coupledSMD':
            self.setup_coupledSMD()

        else:
            raise ValueError("Unknown system type")
        
        self.set_noise(v_d, v_n)

####################################### Helper Functions #######################################
    def _make_A(self, N, k, drag):
        """
        Private helper: Build the dynamics matrix for the coupled SMD system.
        """
        A = np.zeros((2 * (N + 1), 2 * (N + 1)))
        for i in range(N + 1):
            A[i, N + 1 + i] = 1
            if i < N:
                A[N + 1 + i, i] = -k[i] - k[i + 1]
                A[N + 1 + i, N + 1 + i] = -drag
                if i - 1 >= 0:
                    A[N + 1 + i, i - 1] = k[i]
                A[N + 1 + i, i + 1] = k[i + 1]

        A[N + 1, -2] = 0
        A[5, :] = 0
        A[11, :] = 0
        return A
    
    def set_noise(self, V_d, V_n):
        """Set noise as scalar (isotropic) or array (per-observable).

        Parameters
        ----------
        V_d : float or array-like
            Observation noise variance(s).
            - float: same variance for every observable.
            - 1-D array of length y_k: per-observable variances.
        V_n : float or array-like
            Process noise variance(s).
            - float: same variance for every state.
            - 1-D array of length x_k: per-state variances.
        """
        # ── Observation noise ─────────────────────────────────────────────────
        V_d = np.atleast_1d(np.asarray(V_d, dtype=float))
        if V_d.ndim == 1 and V_d.size == 1:
            self.V_d = float(V_d.item()) * np.eye(self.y_k)
        elif V_d.ndim == 1 and V_d.size == self.y_k:
            self.V_d = np.diag(V_d)
        else:
            raise ValueError(
                f"V_d must be a scalar or length-{self.y_k} array, got shape {V_d.shape}")

        # Store per-observable variances for easy reference
        self.obs_noise_var = np.diag(self.V_d).copy()   # (y_k,)

        # ── Process noise ─────────────────────────────────────────────────────
        V_n = np.atleast_1d(np.asarray(V_n, dtype=float))
        if V_n.ndim == 1 and V_n.size == 1:
            self.V_n = float(V_n.item()) * np.eye(self.x_k)
        elif V_n.ndim == 1 and V_n.size == self.x_k:
            self.V_n = np.diag(V_n)
        else:
            raise ValueError(
                f"V_n must be a scalar or length-{self.x_k} array, got shape {V_n.shape}")

####################################### f functions for system dynamics #######################################

    def Masses_2D_f(self, x, u):
        return self.A @ x + self.B @ u
    
    def SMD_f(self, x, u=0):
        x_dot = np.dot(self.A, x) + np.dot(self.B, u).T
        return x_dot

    def coupledSMD_f(self, x, u=0):
        x_dot = np.dot(self.A, x) + np.dot(self.B, u).flatten()
        return x_dot

####################################### g functions for observation #######################################

    def basic_g(self, x):
        """Observation function: full-state observation."""
        return self.C @ x

####################################### System Setups #######################################
    
    def setup_Masses_2D(self, N = 2, drag = 0.5):
        """
        Revised drones dynamics for N drones.
        Each drone has a state vector of length 4: [x, y, vx, vy].
        Overall state dimension: 4*N; control input dimension: 2*N.
        """
        self.n = N
        self.x_k = 4 * N
        self.y_k = 4 * N
        self.u_k = 2 * N
        self.z_k = 4*N

        self.x0 = np.zeros(self.x_k)
        self.x0_lin = self.x0
        self.u_harmless = np.zeros(self.u_k)

        self.A = np.zeros((self.x_k, self.x_k))
        self.A[:2 * N, 2 * N:] = np.eye(2 * N)
        self.A[2 * N:, 2 * N:] = -drag * np.eye(2 * N)

        self.B = np.zeros((self.x_k, self.u_k))
        self.B[self.u_k:, :] = np.eye(self.u_k)

        self.C = np.eye(self.y_k)

        self.A_lin = self.A
        self.B_lin = self.B

        self.f = self.Masses_2D_f
        self.g = self.basic_g
    
    def setup_SMD(self):
        m = 1
        k_val = 3
        c_val = 1

        self.x_k = 2
        self.u_k = 1
        self.y_k = 2
        self.z_k = 2

        self.x0 = np.array([5, 0])
        self.x0_lin = self.x0
        self.u_harmless = np.zeros(1)

        self.A = np.array([[     0,      1/m],
                           [-k_val, -c_val/m]])
        self.B = np.array([[0, 1]], dtype=float).T
        self.C = np.eye(self.y_k)
        
        self.A_lin = self.A
        self.B_lin = self.B

        self.f = self.SMD_f
        self.g = self.basic_g

    def setup_coupledSMD(self, N = 5):
        self.N_coupled = N
        self.L_coupled = 10.0
        self.k = 1 + self.rng.exponential(scale=1, size=N+1)
        self.drag = 0.1

        self.x_k = 2 * (N + 1)
        self.u_k = N
        self.y_k = 2 * (N + 1)
        self.z_k = 2*(N + 1)

        self.x0 = np.zeros(2 * (N + 1))
        self.x0[:N+1] = np.linspace(0, self.L_coupled, N + 1)
        self.x0_lin = self.x0
        self.u_harmless = np.zeros(N)

        self.A = self._make_A(N, self.k, self.drag)
        
        self.B = np.zeros((self.x_k, self.u_k))
        self.B[N+1:2*N+1, :] = np.eye(self.u_k)

        self.C = np.eye(self.y_k)

        self.A_lin = self.A.copy()
        self.A_lin[11, 4], self.A_lin[11, 5] = self.k[5], -self.k[5]
        self.B_lin = self.B
        
        self.f = self.coupledSMD_f
        self.g = self.basic_g

######################################################## Plant Step Function ########################################################
    def step(self, x, u=0, dt = 0.001):
        """
        Update the state using a 4th-order Runge-Kutta integration.
        Process and observation noise are added.
        """
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5 * dt * k1, u)
        k3 = self.f(x + 0.5 * dt * k2, u)
        k4 = self.f(x + dt * k3, u)
        x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        x = x + np.sqrt(dt) * self.rng.multivariate_normal(np.zeros(self.x_k), self.V_n)

        if self.system == 'coupledSMD':
            x[5] = 10
            x[11] = 0
            
        y = self.g(x) + np.sqrt(dt) * self.rng.multivariate_normal(np.zeros(self.y_k), self.V_d)
        return x, y
