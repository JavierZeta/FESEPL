"""Free-Energy State Estimator with Precision Learning.

This file is intentionally written as a companion to `Derivation.pdf`.
The computational heart of the estimator is small:

1. Encode the state estimate as a population code:      mu = D r
2. Build frozen channel precisions:                    Pi_y, Pi_mu
3. Relax spikes by the free-energy voltage rule:       v = b - O r
4. Integrate the local shunting drive:                 tau_d d_dot = sgn(phi)c1 - (c0+eps)d
5. Learn precision from pre-fit innovations:           nu = y - mu_prior
6. Reconstruct variances from gain and scale:          q_mu = K S, q_y = (1-K) S

The comments below use two labels:

- MATH: the line is one of the equations from the derivation.
- GUARDRAIL: validation, clipping, shape handling, or numerical safety.

Those guardrails matter for making the code robust, but they are not new
dynamics. If someone wants to reimplement the algorithm, the MATH comments are
the shortest path through the file.
"""

import numpy as np


rng = np.random.default_rng()


class FESEPL:
    """Spiking free-energy state estimator for the matched-channel case.

    The implementation follows the derivation in five blocks.

    Setup
        Store the plant matrices, create the decoder D, and initialise the
        variance/gain state.

    Fast loop
        For one outer time step, hold y, mu_prior, Pi_y, and Pi_mu fixed. The
        population then spikes only when doing so decreases the quadratic
        free energy.

    Exact posterior
        Compute mu_star, the closed-form Bayesian posterior for the same frozen
        precisions. This is a diagnostic target for the spiking approximation,
        not an input to the spike dynamics.

    Slow traces
        Measure the pre-fit innovation nu = y - mu_prior and keep local traces
        z, c0, c1. These are the statistics used for whitening.

    Precision learning
        Store and learn gain K directly from the innovation traces, then
        reconstruct q_mu = K S and q_y = (1 - K) S.

        C = I,  m = n .

    Fast loop
    ---------
    The coding population stores filtered spike counts `r`, and the decoded
    posterior mean is

        mu = D r .

    With frozen channel precisions

        Pi_y = diag(exp(lambda_y))
        Pi_mu = diag(exp(lambda_mu)) ,

    the fast loop minimizes the channel-space free energy

        F = e_y^T Pi_y e_y + e_mu^T Pi_mu e_mu + beta r^T r

    with

        e_y = y - mu
        e_mu = mu_prior - mu .

    Slow loop
    ---------
    The slow learner does not read off posterior residual identities. It
    learns the effective scalar gain and innovation scale from the innovation
    stream itself.

    For each channel the local innovation is

        nu = y - mu_prior .

    The learner keeps only local slow traces

        z   ~ delayed innovation
        c0  ~ innovation power       E[nu^2]
        c1  ~ lag-one covariance     E[nu z] .

    In the scalar matched-channel case, the correct Bayesian gain is the one
    that whitens the innovation sequence. Positive lag-one covariance means the
    gain is too small, and negative lag-one covariance means it is too large.

    So the slow state is parameterized as

        K = learned gain
        S = innovation scale

    and the two variances used by the fast free energy are reconstructed as

        q_mu = K S
        q_y  = (1 - K) S .
    """

    def __init__(
        self,
        system,
        N=None,
        tau=0.1,
        beta=0.0,
        tau_q=1.0,
        tau_lambda=10.0,
        tau_smooth=0.1,
        eta_lambda=1.0,
        kappa_y=0.01,
        kappa_mu=0.01,
        gain_min=0.02,
        gain_max=0.98,
        lambda_y_step_max=0.25,
        lambda_mu_step_max=0.25,
        observation_window_factor=2.0,
        prior_window_factor=1.0,
        n_inner=25,
        max_spike_rounds=4,
        decoder_seed=None,
        observation_variance_guess=None,
        prior_variance_guess=None,
        variance_floor=1e-8,
        lambda_clip=(-20.0, 20.0),
    ):
        # --- Plant and dimensions -------------------------------------------------
        self.system = system
        self.rng = np.random.default_rng(decoder_seed) if decoder_seed is not None else rng

        self.n = system.x_k
        self.m = system.y_k

        # MATH: linear internal model used for the predictive prior.
        #       mu_prior = Phi mu, with Phi ~= I + dt A_int.
        self.A_int = np.array(system.A_lin, dtype=float, copy=True)
        self.C = np.array(system.C, dtype=float, copy=True)

        # --- Numerical parameters and guardrails ---------------------------------
        self.N = max(2 * self.n, 32) if N is None else int(N)
        if self.N < 2 * self.n:
            raise ValueError(f"N must be at least 2 * x_k = {2 * self.n}, got {self.N}")

        self.tau = float(tau)
        self.beta = float(beta)
        self.tau_q = float(tau_q)
        self.tau_lambda = float(tau_lambda)
        # MATH: tau_smooth is the time constant tau_d of the slow shunting
        # drive used by the precision learner.
        self.tau_smooth = float(tau_smooth)
        self.eta_lambda = float(eta_lambda)
        self.kappa_y = float(kappa_y)

        # Compatibility parameters from the older logit-gain learner. The
        # direct-K learner does not use them; gain is now controlled by
        # eta_lambda, tau_lambda, gain_min, and gain_max.
        self.kappa_mu = float(kappa_mu)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.lambda_y_step_max = float(lambda_y_step_max)

        # Compatibility parameter from the older logit-gain learner. The scale
        # learner still uses lambda_y_step_max; direct K is clipped by gain_min
        # and gain_max instead.
        self.lambda_mu_step_max = float(lambda_mu_step_max)
        self.observation_window_factor = float(observation_window_factor)
        self.prior_window_factor = float(prior_window_factor)
        self.n_inner = int(n_inner)
        self.max_spike_rounds = int(max_spike_rounds)
        self.variance_floor = float(variance_floor)
        self.lambda_min = float(lambda_clip[0])
        self.lambda_max = float(lambda_clip[1])
        if self.lambda_y_step_max <= 0.0:
            raise ValueError("lambda_y_step_max must be positive")
        if self.lambda_mu_step_max <= 0.0:
            raise ValueError("lambda_mu_step_max must be positive")
        if self.tau_smooth <= 0.0:
            raise ValueError("tau_smooth must be positive")
        if self.observation_window_factor <= 0.0:
            raise ValueError("observation_window_factor must be positive")
        if self.prior_window_factor <= 0.0:
            raise ValueError("prior_window_factor must be positive")
        if not (0.0 < self.gain_min < self.gain_max < 1.0):
            raise ValueError("gain_min and gain_max must satisfy 0 < gain_min < gain_max < 1")

        # GUARDRAIL: this implementation is exactly the matched-channel case
        # from Derivation sections 2.4 and 10.1.
        self.full_state_observation = (
            self.m == self.n and np.allclose(self.C, np.eye(self.n))
        )
        if not self.full_state_observation:
            raise NotImplementedError(
                "This FESEPL precision learner assumes full-state observation "
                "(C = I, m = n)."
            )

        # --- Slow learner initial state ------------------------------------------
        self.discrete_targets_ready = False

        # GUARDRAIL: q_*_initial are the live starting values used to
        # stress-test recovery. Any slow regularization is allowed to pull only
        # toward this initial estimator guess, never toward an analytically
        # computed target.
        self.q_y_initial = self._make_variance_guess(
            observation_variance_guess,
            self.m,
            np.ones(self.m),
        )
        self.q_mu_initial = self._make_variance_guess(
            prior_variance_guess,
            self.n,
            np.ones(self.n),
        )
        self.q_y0 = self.q_y_initial.copy()
        self.q_mu0 = self.q_mu_initial.copy()

        # --- Decoder and initial gain/scale --------------------------------------
        # MATH: neural code, mu = D r.
        self.D = self._build_decoder()

        # MATH: store log-precision equivalents and direct gain.
        #       K = q_mu / (q_mu + q_y), S = q_mu + q_y.
        self.lambda_y0 = -np.log(self.q_y0)
        self.lambda_mu0 = -np.log(self.q_mu0)
        self.gain0 = np.clip(
            self.q_mu0 / np.maximum(self.q_mu0 + self.q_y0, self.variance_floor),
            self.gain_min,
            self.gain_max,
        )
        self.scale0 = np.clip(self.q_mu0 + self.q_y0, self.variance_floor, None)

        self.reset()

    def _make_variance_guess(self, guess, dim, fallback):
        """Return a positive channel-variance vector.

        This is input hygiene, not estimator dynamics. Scalars, vectors, and
        diagonal covariance matrices are accepted so experiments can be written
        conveniently.
        """
        if guess is None:
            guess = np.array(fallback, dtype=float, copy=True)
        else:
            guess = np.asarray(guess, dtype=float)
            if guess.ndim == 0:
                guess = np.full(dim, float(guess))
            elif guess.ndim == 1:
                if guess.size != dim:
                    raise ValueError(f"Variance guess must have length {dim}, got {guess.size}")
            elif guess.shape == (dim, dim):
                guess = np.diag(guess)
            else:
                raise ValueError(f"Variance guess must be scalar, length-{dim}, or {(dim, dim)}, got {guess.shape}")

        return np.clip(np.array(guess, dtype=float, copy=True), self.variance_floor, None)

    def _build_decoder(self):
        """
        Build the fixed population decoder D.

        MATH: the neural estimate is always `mu = D r`.

        Implementation choice: signed basis columns guarantee every state
        dimension has at least one positive and one negative coding direction.
        Extra columns are random unit directions. The final scale keeps spike
        magnitudes in the same range as the original experiments.

        Match the simple style used in the earlier FESEPL class:
        signed basis columns first, random unit columns after that, then shuffle.
        """
        D = np.zeros((self.n, self.N))

        for i in range(self.n):
            D[i, 2 * i] = 1.0
            D[i, 2 * i + 1] = -1.0

        for i in range(2 * self.n, self.N):
            col = self.rng.normal(0.0, 1.0, self.n)
            norm = np.linalg.norm(col)
            if norm > 0.0:
                col = col / norm
            D[:, i] = col

        shuffled = self.rng.permutation(self.N)

        if getattr(self.system, "system", "") == "SMD":
            scale = 1.0 * self.N
        else:
            scale = 0.1 * self.N

        return D[:, shuffled] / scale

    def set_precisions(self, pi_y, pi_mu):
        """
        Replace the live channel precisions directly.

        This is mainly used by diagnostics and frozen-precision experiments.
        The corresponding variance estimates are updated to stay consistent.
        """
        pi_y = np.asarray(pi_y, dtype=float)
        pi_mu = np.asarray(pi_mu, dtype=float)
        if pi_y.shape != (self.n,) or pi_mu.shape != (self.n,):
            raise ValueError(
                f"pi_y and pi_mu must both have shape ({self.n},)"
            )

        pi_y = np.clip(pi_y, self.variance_floor, None)
        pi_mu = np.clip(pi_mu, self.variance_floor, None)
        self.lambda_y = np.log(pi_y)
        self.lambda_mu = np.log(pi_mu)
        self.q_y = 1.0 / pi_y
        self.q_mu = 1.0 / pi_mu
        gain = np.clip(
            self.q_mu / np.maximum(self.q_mu + self.q_y, self.variance_floor),
            self.gain_min,
            self.gain_max,
        )
        self.current_gain = gain
        self.scale = np.clip(self.q_mu + self.q_y, self.variance_floor, None)
        self._refresh_precisions()

    def _reset_precision_traces(self):
        """Reset the local innovation-whitening traces."""
        self.z = np.zeros(self.n)
        self.c0 = np.zeros(self.n)
        self.c1 = np.zeros(self.n)

    def reset(self):
        """Reset all dynamic state without rebuilding the decoder or parameters."""
        # --- State estimates and observations ------------------------------------
        self.mu = np.zeros(self.n)
        self.mu_prior = np.zeros(self.n)
        self.mu_star = np.zeros(self.n)
        self.y = np.zeros(self.m)

        # --- Spike population state ----------------------------------------------
        # MATH: r stores filtered spike counts, v stores the membrane voltage.
        self.r = np.zeros(self.N)
        self.v = np.zeros(self.N)
        self.spike_totals = np.zeros(self.N)

        # MATH: v = bias - O r. `window_O` is the O used inside the current
        # frozen fast-loop window.
        self.bias = np.zeros(self.N)
        self.O = np.zeros((self.N, self.N))
        self.window_O = None

        # --- Fast-loop errors -----------------------------------------------------
        # MATH: e_y = y - mu, e_mu = mu_prior - mu.
        self.e_y = np.zeros(self.m)
        self.e_mu = np.zeros(self.n)
        self.e_y_weighted = np.zeros(self.m)
        self.e_mu_weighted = np.zeros(self.n)

        # --- Innovation and diagnostics ------------------------------------------
        # MATH: nu = y - mu_prior is measured before posterior correction.
        self.nu_y = np.zeros(self.m)
        self.nu_mu = np.zeros(self.n)
        self.eps_y = np.zeros(self.m)
        self.eps_mu = np.zeros(self.n)
        self.delta_mu = np.zeros(self.n)

        # --- Slow learner state ---------------------------------------------------
        # MATH: current local variance estimates. These define the fast-loop
        # precisions and are updated only after the innovation traces are updated.
        self.q_y = self.q_y_initial.copy()
        self.q_mu = self.q_mu_initial.copy()
        self.current_gain = self.gain0.copy()
        self.scale = self.scale0.copy()

        # MATH: pi_y = 1/q_y and pi_mu = 1/q_mu, represented via lambda = log(pi).
        self.lambda_y = -np.log(self.q_y)
        self.lambda_mu = -np.log(self.q_mu)
        self.pi_y = np.zeros(self.n)
        self.pi_mu = np.zeros(self.n)
        self.posterior_variance = np.zeros(self.n)
        self.settled_gate = 0.0
        self._reset_precision_traces()
        self.q_y_target = np.zeros(self.n)
        self.q_mu_target = np.zeros(self.n)
        self.k_target = np.zeros(self.n)
        self.s_target = np.zeros(self.n)
        self.d = np.zeros(self.n)
        self.d_raw = np.zeros(self.n)
        self.d_smooth = self.d.copy()
        self.gain_drive = np.zeros(self.n)
        self.window_activity = 0.0
        self.q_y_floor_hits = 0
        self.q_mu_floor_hits = 0
        self.T_base = np.zeros(self.N)
        self.threshold = np.zeros(self.N)
        self.Phi = np.eye(self.n)

        self.free_energy = 0.0
        self.free_energy_start = 0.0
        self.free_energy_end = 0.0

        self.step_count = 0
        self.initialised = False

    def setup(self, y0=None, mu0=None):
        """Initialise the posterior state estimate before the first update.

        Most of this method is compatibility/setup. The key mathematical line is
        the encoding `r = pinv(D) mu`, which chooses spike counts whose decoded
        state starts at the requested initial posterior mean.
        """
        if mu0 is None:
            if y0 is None:
                mu0 = np.array(self.system.x0_lin, dtype=float, copy=True)
            else:
                mu0 = np.linalg.lstsq(self.C, y0, rcond=None)[0]
        else:
            mu0 = np.array(mu0, dtype=float, copy=True)

        if y0 is None:
            y0 = self.C @ mu0
        else:
            y0 = np.array(y0, dtype=float, copy=True)

        self.mu = mu0.copy()
        self.mu_prior = mu0.copy()
        self.y = y0.copy()

        # MATH: initialise the neural code so that mu = D r at time zero.
        self.r = np.linalg.pinv(self.D) @ self.mu

        self._refresh_precisions()

        # MATH: bias b = D^T (Pi_y y + Pi_mu mu_prior).
        self.bias = self.D.T @ (self.pi_y * self.y + self.pi_mu * self.mu_prior)
        self.window_O = self.O.copy()

        # MATH: algebraic voltage v = b - O r.
        self.v = self.bias - self.window_O @ self.r
        self._refresh_fast_state(self.y, self.mu_prior)
        self._compute_exact_posterior(self.y, self.mu_prior)

        self.initialised = True
        self.step_count = 0

    def _make_phi(self, dt):
        """Discrete state-transition approximation: Phi ~= I + dt A_int."""
        return np.eye(self.n) + dt * self.A_int

    def _predict_prior(self, dt):
        """MATH: predictive prior, mu_prior = Phi mu."""
        self.Phi = self._make_phi(dt)
        return self.Phi @ self.mu

    def _ensure_discrete_targets(self, dt):
        """Freeze the initial regularization centre once the estimator is live."""
        if self.discrete_targets_ready:
            return

        # GUARDRAIL: freeze the regularization centre at the estimator's actual initial
        # guess. No hidden model-derived target is used here.
        self.q_y0 = self.q_y_initial.copy()
        self.q_mu0 = self.q_mu_initial.copy()
        self.lambda_y0 = -np.log(self.q_y0)
        self.lambda_mu0 = -np.log(self.q_mu0)
        self.gain0 = np.clip(
            self.q_mu0 / np.maximum(self.q_mu0 + self.q_y0, self.variance_floor),
            self.gain_min,
            self.gain_max,
        )
        self.scale0 = np.clip(self.q_mu0 + self.q_y0, self.variance_floor, None)
        self._reset_precision_traces()
        self.discrete_targets_ready = True

    def _refresh_precisions(self):
        """
        Rebuild the live channel precisions and the recurrent curvature matrix.

        With `C = I`, the coding-neuron drive can always be written as

            v* = b - O r

        where `O = D^T (Pi_y + Pi_mu) D`. The fast membrane dynamics inside one
            frozen window tracks exactly this algebraic voltage.
        """
        # MATH: K is stored directly and clipped to the permitted gain interval.
        self.current_gain = np.clip(self.current_gain, self.gain_min, self.gain_max)

        # MATH: reconstruct channel variances from gain and total scale:
        #       q_mu = K S, q_y = (1 - K) S.
        self.scale = np.clip(self.scale, self.variance_floor, None)
        self.q_mu = np.clip(self.current_gain * self.scale, self.variance_floor, None)
        self.q_y = np.clip((1.0 - self.current_gain) * self.scale, self.variance_floor, None)

        # MATH: precisions are inverse variances, represented as exp(lambda).
        self.lambda_y = np.clip(-np.log(self.q_y), self.lambda_min, self.lambda_max)
        self.lambda_mu = np.clip(-np.log(self.q_mu), self.lambda_min, self.lambda_max)
        self.pi_y = np.exp(self.lambda_y)
        self.pi_mu = np.exp(self.lambda_mu)

        # MATH: exact posterior variance for C = I and diagonal precisions.
        self.posterior_variance = 1.0 / np.maximum(
            self.pi_y + self.pi_mu,
            self.variance_floor,
        )

        # MATH: recurrent curvature matrix, O = D^T diag(pi_y + pi_mu) D.
        self.O = self.D.T @ ((self.pi_y + self.pi_mu)[:, None] * self.D)

        # MATH: base spike threshold T_base = 0.5 diag(O).
        self.T_base = 0.5 * np.diag(self.O)

    def _refresh_fast_state(self, y, mu_prior):
        """Recompute every fast-loop quantity derived from r, y, and mu_prior."""
        # MATH: decode the population state, mu = D r.
        self.mu = self.D @ self.r

        # MATH: matched-channel prediction errors.
        self.e_y = y - self.mu
        self.e_mu = mu_prior - self.mu

        # MATH: precision-weighted prediction errors.
        self.e_y_weighted = self.pi_y * self.e_y
        self.e_mu_weighted = self.pi_mu * self.e_mu

        self._refresh_threshold()

        # MATH: fast-loop free energy:
        #       F = e_y^T Pi_y e_y + e_mu^T Pi_mu e_mu + beta r^T r.
        self.free_energy = (
            self.e_y @ self.e_y_weighted
            + self.e_mu @ self.e_mu_weighted
            + self.beta * (self.r @ self.r)
        )

    def _refresh_threshold(self):
        """Refresh only the spike threshold needed inside the inner loop."""
        # MATH: spike threshold, T_i = 0.5 O_ii + beta (r_i + 0.5).
        self.threshold = self.T_base + self.beta * (self.r + 0.5)

    def _compute_exact_posterior(self, y, mu_prior):
        """Closed-form Bayesian posterior for the same frozen precisions.

        This does not drive the spiking dynamics; it is the comparison target
        `mu_star` used in diagnostics.
        """
        # MATH: with C = I and diagonal precisions,
        #       mu_star = (pi_y y + pi_mu mu_prior) / (pi_y + pi_mu).
        denom = self.pi_y + self.pi_mu
        information = self.pi_y * y + self.pi_mu * mu_prior
        self.mu_star = information / np.maximum(denom, self.variance_floor)

    def _record_prefit_snapshots(self, y, mu_prior):
        """Record pre-relaxation quantities used by diagnostics and learning."""
        r_start = self.r.copy()
        mu_start = self.D @ r_start

        e_y_start = y - mu_start
        e_mu_start = mu_prior - mu_start

        # MATH: innovation is measured before posterior correction.
        self.nu_y = y - mu_prior
        self.nu_mu = self.nu_y.copy()

        # MATH: free energy at the start of the fast relaxation window.
        self.free_energy_start = (
            e_y_start @ (self.pi_y * e_y_start)
            + e_mu_start @ (self.pi_mu * e_mu_start)
            + self.beta * (r_start @ r_start)
        )

    def _run_inference(self, y, mu_prior, dt):
        """
        Fast spiking relaxation.

        The coding state follows

            r_dot = -tau r + s

        and the membrane voltage follows the exact dynamic equivalent of the
        algebraic free-energy drive

            v* = b - O r

        inside one frozen window:

            v_dot = -tau v + tau b - O s .

        At the start of a new outer step, the bias and curvature can jump
        because `y`, `mu_prior`, or the learned precisions changed. The exact
        jump that preserves the algebraic free-energy voltage is

            v <- v + (b_new - b_old) - (O_new - O_old) r .

        With that jump plus the inner leaky dynamics below, the voltage remains
        exactly equivalent to the old algebraic assignment while now being a
        proper membrane state.
        """
        self._record_prefit_snapshots(y, mu_prior)
        self.spike_totals.fill(0.0)

        # MATH: bias b = D^T (Pi_y y + Pi_mu mu_prior).
        new_bias = self.D.T @ (self.pi_y * y + self.pi_mu * mu_prior)
        new_O = self.O

        # MATH: between-step jump preserves v = b - O r when b or O changes.
        if self.window_O is None:
            self.v = new_bias - new_O @ self.r
        else:
            self.v = self.v + (new_bias - self.bias) - (new_O - self.window_O) @ self.r
        self.bias = new_bias
        self.window_O = new_O

        inner_dt = dt / max(self.n_inner, 1)

        for _ in range(self.n_inner):
            # MATH: leaky spike-count dynamics, r_dot = -tau r + s.
            self.r = self.r + inner_dt * (-self.tau * self.r)

            # MATH: membrane dynamics between spikes, v_dot = -tau v + tau b.
            # The instantaneous -O s term is applied below when spikes occur.
            self.v = self.v + inner_dt * (-self.tau * self.v + self.tau * self.bias)
            self._refresh_threshold()

            for _ in range(self.max_spike_rounds):
                # MATH: spike rule from Delta F_i < 0, neuron fires if v_i > T_i.
                spiking = self.v > self.threshold
                if not np.any(spiking):
                    break

                spike_vector = spiking.astype(float)
                self.spike_totals += spike_vector

                # MATH: spike update, r <- r + s and v <- v - O s.
                self.r = self.r + spike_vector
                self.v = self.v - self.window_O @ spike_vector
                self._refresh_threshold()

        self._refresh_fast_state(y, mu_prior)
        self.eps_y = self.e_y.copy()
        self.eps_mu = self.e_mu.copy()
        self.delta_mu = self.mu - mu_prior
        self.free_energy_end = self.free_energy

        # DIAGNOSTIC: strict fast-loop settling flag. It is observed, but it no
        # longer gates the slow learner.
        self.settled_gate = float(
            np.max(self.v - self.threshold) <= 1e-10
            and self.free_energy_end <= self.free_energy_start + 1e-10
        )

    def _update_innovation_traces(self, dt):
        """
        Update the local traces used by the innovation-whitening learner.

        Each channel keeps:

            z   : one-step delayed innovation copy
            c0  : innovation power trace
            c1  : lag-one innovation covariance trace
        """
        # GUARDRAIL: use positive finite time constants even if a config passes
        # a very small value.
        trace_tau = max(self.tau_q * self.observation_window_factor, 1e-12)
        trace_step = float(dt) / trace_tau
        z_prev = self.z.copy()

        # MATH: power trace, tau c0_dot = -c0 + nu^2.
        self.c0 += trace_step * ((self.nu_y ** 2) - self.c0)

        # MATH: lag-one trace, tau c1_dot = -c1 + nu z_previous.
        self.c1 += trace_step * ((self.nu_y * z_prev) - self.c1)

        # MATH: delay trace stores the current innovation for the next step.
        self.z = self.nu_y.copy()

        # DIAGNOSTIC: normalized activity in settled windows only. The gate is
        # observed here, but it no longer controls the estimator's trace update.
        self.window_activity = self.settled_gate * float(np.sum(self.spike_totals)) / max(self.N * self.n_inner, 1)

    def _update_precisions_from_whiteness(self, dt):
        """
        Learn the effective Bayesian gain by whitening the innovation stream,
        then reconstruct the two variances from the learned gain and scale.
        """
        # GUARDRAIL: positive finite slow-learning step.
        slow_tau = max(self.tau_lambda * self.prior_window_factor, 1e-12)
        slow_step = float(dt) / slow_tau

        # MATH: signed covariance excites the local metaplastic drive.
        #       d_raw is kept as a diagnostic numerator:
        #       d_raw,k = sign(phi_k) c1_k.
        phi_diag = np.diag(self.Phi)
        phi_sign = np.where(phi_diag >= 0.0, 1.0, -1.0)
        self.d_raw = phi_sign * self.c1

        # MATH: local shunting-normalized astrocyte drive,
        #       tau_d d_dot_k = sign(phi_k)c1_k - (c0_k + eps)d_k.
        #
        # At equilibrium this has the same fixed point as the old normalized
        # drive, d_k = sign(phi_k)c1_k / (c0_k + eps), but the normalization is
        # expressed as local shunting instead of an explicit division plus LPF.
        shunt = self.c0 + self.variance_floor
        alpha_drive = float(dt) / self.tau_smooth
        self.d = self.d + alpha_drive * (self.d_raw - shunt * self.d)
        self.d_smooth = self.d.copy()
        self.gain_drive = self.d.copy()

        # MATH: direct gain update, K <- clip(K + eta_K d, K_min, K_max).
        # Here eta_K keeps the existing outer-loop time scaling:
        # eta_K = dt / tau_lambda * eta_lambda.
        gain_step = slow_step * self.eta_lambda * self.d
        self.current_gain = np.clip(
            self.current_gain + gain_step,
            self.gain_min,
            self.gain_max,
        )
        self.k_target = self.current_gain.copy()

        # MATH: scale target is innovation power, S_target = c0.
        scale_target = np.clip(self.c0, self.variance_floor, None)

        # MATH: S_dot = (S_target - S) / tau - kappa_y (S - S0).
        scale_new = self.scale + slow_step * (scale_target - self.scale)
        scale_new += self.kappa_y * slow_step * (self.scale0 - scale_new)
        scale_new = np.clip(scale_new, self.variance_floor, None)

        # GUARDRAIL: cap one-step scale movement in log-space.
        log_scale = np.log(np.maximum(self.scale, self.variance_floor))
        log_scale_new = np.log(scale_new)
        log_scale_step = np.clip(
            log_scale_new - log_scale,
            -self.lambda_y_step_max,
            self.lambda_y_step_max,
        )
        self.scale = np.exp(log_scale + log_scale_step)
        self.s_target = scale_target

        # Diagnostic: target variances reconstructed from target gain and scale.
        self.q_mu_target = np.clip(self.k_target * self.s_target, self.variance_floor, None)
        self.q_y_target = np.clip((1.0 - self.k_target) * self.s_target, self.variance_floor, None)

        # DIAGNOSTIC: count channels that hit the variance floor.
        self.q_y_floor_hits += int(np.count_nonzero(self.q_y_target <= 1.01 * self.variance_floor))
        self.q_mu_floor_hits += int(np.count_nonzero(self.q_mu_target <= 1.01 * self.variance_floor))

        # MATH: push the new gain/scale back into q_y, q_mu, pi_y, pi_mu, O, T.
        self._refresh_precisions()

    def update(self, y, dt):
        """
        One outer estimator step.

        Step logic:
        1. Predict the prior state.
        2. Relax the coding population with frozen channel precisions.
        3. Update local innovation traces.
        4. Learn gain from innovation whitening and scale from innovation power.
        """
        y = np.array(y, dtype=float, copy=True)

        self._ensure_discrete_targets(dt)

        if not self.initialised:
            self.setup(y0=y)

        self.y = y

        if self.step_count == 0:
            self.mu_prior = self.mu.copy()
            self.Phi = self._make_phi(dt)
        else:
            # MATH: outer-step prediction, mu_prior = Phi mu.
            self.mu_prior = self._predict_prior(dt)

        # MATH: relax the spiking code against fixed y, mu_prior, Pi_y, Pi_mu.
        self._run_inference(self.y, self.mu_prior, dt)

        # DIAGNOSTIC: keep mu_star aligned with the same frozen precisions that produced the
        # current spiking posterior mu.
        self._compute_exact_posterior(self.y, self.mu_prior)

        # MATH: learn from the pre-fit innovation stream.
        self._update_innovation_traces(dt)
        self._update_precisions_from_whiteness(dt)

        self.step_count += 1
        return self.mu.copy()
