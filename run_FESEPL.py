import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MPL_DIR = ROOT / "artifacts" / "mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from KalmanFilterClass import KalmanFilter
from PlantClass import Plant
from FESEPL import FESEPL


@dataclass
class RunConfig:
    name: str
    mode: str = "learn"

    system: str = "SMD"
    dt: float = 0.005
    total_time: float = 15.0
    seed: int = 0

    plant_process_noise: float = 0.0
    observation_noise: float = 1.0
    internal_process_noise: float | None = None

    initial_state_mode: str = "zero"
    kalman_initial_covariance: float = 10.0

    observation_variance_guess: float = 1e-2
    prior_variance_guess: float = 2e-1

    N: int = 64
    tau: float = 0.1
    beta: float = 0.05
    tau_q: float = 5.0
    tau_lambda: float = 80.0
    eta_lambda: float = 0.1
    kappa_y: float = 0.0
    kappa_mu: float = 0.0
    gain_min: float = 0.02
    gain_max: float = 0.98
    lambda_y_step_max: float = 0.25
    lambda_mu_step_max: float = 0.25
    observation_window_factor: float = 2.0
    prior_window_factor: float = 2.0
    n_inner: int = 150
    max_spike_rounds: int = 32
    decoder_seed: int = 5

    save_dir: str = "outputs"


def build_plant(config):
    return Plant(
        system=config.system,
        v_n=config.plant_process_noise / config.dt,
        v_d=config.observation_noise / config.dt,
        seed=config.seed,
    )


def build_estimator(plant, config):
    return FESEPL(
        plant,
        N=config.N,
        tau=config.tau,
        beta=config.beta,
        tau_q=config.tau_q,
        tau_lambda=config.tau_lambda,
        eta_lambda=config.eta_lambda,
        kappa_y=config.kappa_y,
        kappa_mu=config.kappa_mu,
        gain_min=config.gain_min,
        gain_max=config.gain_max,
        lambda_y_step_max=config.lambda_y_step_max,
        lambda_mu_step_max=config.lambda_mu_step_max,
        observation_window_factor=config.observation_window_factor,
        prior_window_factor=config.prior_window_factor,
        n_inner=config.n_inner,
        max_spike_rounds=config.max_spike_rounds,
        decoder_seed=config.decoder_seed,
        observation_variance_guess=config.observation_variance_guess,
        prior_variance_guess=config.prior_variance_guess,
    )


def build_kalman(plant, config):
    process_noise = (
        config.plant_process_noise
        if config.internal_process_noise is None
        else config.internal_process_noise
    )
    return KalmanFilter(
        plant,
        dt=config.dt,
        process_noise=process_noise / config.dt,
        observation_noise=config.observation_noise / config.dt,
    )


def make_initial_state_guess(plant, config):
    if config.initial_state_mode == "zero":
        return np.zeros(plant.x_k)
    if config.initial_state_mode == "x0":
        return plant.x0.copy()
    raise ValueError(f"Unknown initial_state_mode: {config.initial_state_mode}")


def compute_true_observation_channel_stats(plant, dt, floor=1e-12):
    q_y = np.clip(np.diag(dt * plant.V_d), floor, None)
    pi_y = 1.0 / q_y
    return q_y, pi_y


def compute_true_prior_channel_stats(kalman, floor=1e-12):
    q_mu = np.clip(np.diag(kalman.P_prior), floor, None)
    pi_mu = 1.0 / q_mu
    return q_mu, pi_mu


def kalman_correct_from_prior(kalman, y):
    y = np.array(y, dtype=float, copy=True)
    kalman.innovation = y - kalman.C @ kalman.x_prior
    kalman.S = kalman.C @ kalman.P_prior @ kalman.C.T + kalman.R_d
    kalman.K = kalman.P_prior @ kalman.C.T @ np.linalg.inv(kalman.S)
    kalman.x = kalman.x_prior + kalman.K @ kalman.innovation

    I = np.eye(kalman.n)
    KC = kalman.K @ kalman.C
    kalman.P = (I - KC) @ kalman.P_prior @ (I - KC).T + kalman.K @ kalman.R_d @ kalman.K.T
    return kalman.x.copy()


def frozen_true_precision_step(estimator, y, dt, true_pi_y, true_pi_mu):
    y = np.array(y, dtype=float, copy=True)

    if not estimator.initialised:
        estimator.setup(y0=y)

    estimator.y = y

    if estimator.step_count == 0:
        estimator.mu_prior = estimator.mu.copy()
        estimator.Phi = estimator._make_phi(dt)
    else:
        estimator.mu_prior = estimator._predict_prior(dt)

    estimator.set_precisions(true_pi_y, true_pi_mu)

    estimator._run_inference(estimator.y, estimator.mu_prior, dt)
    estimator._refresh_fast_state(estimator.y, estimator.mu_prior)
    estimator._compute_exact_posterior(estimator.y, estimator.mu_prior)

    estimator.step_count += 1
    return estimator.mu.copy()


def allocate_history(steps, plant, estimator):
    return {
        "t": np.zeros(steps + 1),
        "x": np.zeros((steps + 1, plant.x_k)),
        "y": np.zeros((steps + 1, plant.y_k)),
        "sfec_mu": np.zeros((steps + 1, plant.x_k)),
        "sfec_mu_prior": np.zeros((steps + 1, plant.x_k)),
        "sfec_mu_star": np.zeros((steps + 1, plant.x_k)),
        "kalman_mu": np.zeros((steps + 1, plant.x_k)),
        "kalman_mu_prior": np.zeros((steps + 1, plant.x_k)),
        "sfec_true_mse": np.zeros(steps + 1),
        "mu_star_true_mse": np.zeros(steps + 1),
        "kalman_true_mse": np.zeros(steps + 1),
        "sfec_mu_star_mse": np.zeros(steps + 1),
        "sfec_kalman_mse": np.zeros(steps + 1),
        "mu_star_kalman_mse": np.zeros(steps + 1),
        "spike_count": np.zeros(steps + 1),
        "active_fraction": np.zeros(steps + 1),
        "free_energy_start": np.zeros(steps + 1),
        "free_energy_end": np.zeros(steps + 1),
        "free_energy_drop": np.zeros(steps + 1),
        "nu_y_norm": np.zeros(steps + 1),
        "eps_y_norm": np.zeros(steps + 1),
        "nu_mu_norm": np.zeros(steps + 1),
        "eps_mu_norm": np.zeros(steps + 1),
        "delta_mu_norm": np.zeros(steps + 1),
        "pi_y_mean": np.zeros(steps + 1),
        "pi_mu_mean": np.zeros(steps + 1),
        "q_y_mean": np.zeros(steps + 1),
        "q_mu_mean": np.zeros(steps + 1),
        "q_y_target_mean": np.zeros(steps + 1),
        "q_mu_target_mean": np.zeros(steps + 1),
        "k_target_mean": np.zeros(steps + 1),
        "current_gain_mean": np.zeros(steps + 1),
        "true_gain_mean": np.zeros(steps + 1),
        "s_target_mean": np.zeros(steps + 1),
        "window_activity": np.zeros(steps + 1),
        "q_y_min": np.zeros(steps + 1),
        "q_mu_min": np.zeros(steps + 1),
        "true_q_y_min": np.zeros(steps + 1),
        "true_q_mu_min": np.zeros(steps + 1),
        "q_y_floor_hits": np.zeros(steps + 1),
        "q_mu_floor_hits": np.zeros(steps + 1),
        "true_pi_y_mean": np.zeros(steps + 1),
        "true_pi_mu_mean": np.zeros(steps + 1),
        "true_q_y_mean": np.zeros(steps + 1),
        "true_q_mu_mean": np.zeros(steps + 1),
        "ratio_mean": np.zeros(steps + 1),
        "true_ratio_mean": np.zeros(steps + 1),
        "q_ratio_mean": np.zeros(steps + 1),
        "ratio_consistency_error": np.zeros(steps + 1),
    }


def record_step(history, idx, time, x, y, estimator, kalman, true_q_y, true_pi_y, true_q_mu, true_pi_mu):
    history["t"][idx] = time
    history["x"][idx] = x
    history["y"][idx] = y
    history["sfec_mu"][idx] = estimator.mu
    history["sfec_mu_prior"][idx] = estimator.mu_prior
    history["sfec_mu_star"][idx] = estimator.mu_star
    history["kalman_mu"][idx] = kalman.x
    history["kalman_mu_prior"][idx] = kalman.x_prior

    history["sfec_true_mse"][idx] = np.mean((estimator.mu - x) ** 2)
    history["mu_star_true_mse"][idx] = np.mean((estimator.mu_star - x) ** 2)
    history["kalman_true_mse"][idx] = np.mean((kalman.x - x) ** 2)
    history["sfec_mu_star_mse"][idx] = np.mean((estimator.mu - estimator.mu_star) ** 2)
    history["sfec_kalman_mse"][idx] = np.mean((estimator.mu - kalman.x) ** 2)
    history["mu_star_kalman_mse"][idx] = np.mean((estimator.mu_star - kalman.x) ** 2)

    history["spike_count"][idx] = estimator.spike_totals.sum()
    history["active_fraction"][idx] = np.mean(estimator.spike_totals > 0.0)

    history["free_energy_start"][idx] = estimator.free_energy_start
    history["free_energy_end"][idx] = estimator.free_energy_end
    history["free_energy_drop"][idx] = estimator.free_energy_start - estimator.free_energy_end

    history["nu_y_norm"][idx] = np.linalg.norm(estimator.nu_y)
    history["eps_y_norm"][idx] = np.linalg.norm(estimator.eps_y)
    history["nu_mu_norm"][idx] = np.linalg.norm(estimator.nu_mu)
    history["eps_mu_norm"][idx] = np.linalg.norm(estimator.eps_mu)
    history["delta_mu_norm"][idx] = np.linalg.norm(estimator.delta_mu)

    history["pi_y_mean"][idx] = estimator.pi_y.mean()
    history["pi_mu_mean"][idx] = estimator.pi_mu.mean()
    history["q_y_mean"][idx] = estimator.q_y.mean()
    history["q_mu_mean"][idx] = estimator.q_mu.mean()
    history["q_y_target_mean"][idx] = estimator.q_y_target.mean()
    history["q_mu_target_mean"][idx] = estimator.q_mu_target.mean()
    history["k_target_mean"][idx] = estimator.k_target.mean()
    history["current_gain_mean"][idx] = estimator.current_gain.mean()
    history["true_gain_mean"][idx] = np.mean(true_q_mu / np.maximum(true_q_mu + true_q_y, estimator.variance_floor))
    history["s_target_mean"][idx] = estimator.s_target.mean()
    history["window_activity"][idx] = estimator.window_activity
    history["q_y_min"][idx] = estimator.q_y.min()
    history["q_mu_min"][idx] = estimator.q_mu.min()
    history["true_q_y_min"][idx] = true_q_y.min()
    history["true_q_mu_min"][idx] = true_q_mu.min()
    history["q_y_floor_hits"][idx] = estimator.q_y_floor_hits
    history["q_mu_floor_hits"][idx] = estimator.q_mu_floor_hits
    history["true_pi_y_mean"][idx] = true_pi_y.mean()
    history["true_pi_mu_mean"][idx] = true_pi_mu.mean()
    history["true_q_y_mean"][idx] = true_q_y.mean()
    history["true_q_mu_mean"][idx] = true_q_mu.mean()
    history["ratio_mean"][idx] = np.mean(
        estimator.pi_mu / np.maximum(estimator.pi_y, estimator.variance_floor)
    )
    history["true_ratio_mean"][idx] = np.mean(
        true_pi_mu / np.maximum(true_pi_y, estimator.variance_floor)
    )
    history["q_ratio_mean"][idx] = np.mean(
        estimator.q_y / np.maximum(estimator.q_mu, estimator.variance_floor)
    )
    history["ratio_consistency_error"][idx] = np.mean(
        np.abs(
            estimator.pi_mu / np.maximum(estimator.pi_y, estimator.variance_floor)
            - estimator.q_y / np.maximum(estimator.q_mu, estimator.variance_floor)
        )
    )


def run_experiment(config):
    plant = build_plant(config)
    estimator = build_estimator(plant, config)
    kalman = build_kalman(plant, config)

    steps = int(config.total_time / config.dt)
    history = allocate_history(steps, plant, estimator)

    x = plant.x0.copy()
    y0 = plant.C @ x
    u = plant.u_harmless.copy()
    x_guess0 = make_initial_state_guess(plant, config)

    estimator.setup(y0=y0, mu0=x_guess0)
    kalman.setup(x0=x_guess0, P0=config.kalman_initial_covariance * np.eye(plant.x_k))

    true_q_y, true_pi_y = compute_true_observation_channel_stats(plant, config.dt)
    true_q_mu0 = np.clip(np.diag(kalman.P_prior), estimator.variance_floor, None)
    true_pi_mu0 = 1.0 / true_q_mu0
    record_step(history, 0, 0.0, x, y0, estimator, kalman, true_q_y, true_pi_y, true_q_mu0, true_pi_mu0)

    for k in range(steps):
        x, y = plant.step(x, u=u, dt=config.dt)

        if config.mode == "freeze_true":
            kalman.predict()
            true_q_mu, true_pi_mu = compute_true_prior_channel_stats(kalman, estimator.variance_floor)
            frozen_true_precision_step(estimator, y, config.dt, true_pi_y, true_pi_mu)
            kalman_correct_from_prior(kalman, y)
        elif config.mode == "learn":
            estimator.update(y, config.dt)
            kalman.update(y)
            true_q_mu, true_pi_mu = compute_true_prior_channel_stats(kalman, estimator.variance_floor)
        else:
            raise ValueError(f"Unknown mode: {config.mode}")

        record_step(
            history,
            k + 1,
            (k + 1) * config.dt,
            x,
            y,
            estimator,
            kalman,
            true_q_y,
            true_pi_y,
            true_q_mu,
            true_pi_mu,
        )

    return plant, estimator, kalman, history


def summarize_experiment(history, config):
    idx = slice(1, None)
    total_time = float(history["t"][-1])
    late_start_time = max(10.0, total_time - 10.0)
    if late_start_time >= total_time:
        late_start_time = max(float(history["t"][1]), 0.5 * total_time)
    late_mask = history["t"] >= late_start_time
    if np.count_nonzero(late_mask) <= 1:
        late_mask = np.zeros_like(history["t"], dtype=bool)
        late_mask[1:] = True

    ratio = history["ratio_mean"][idx]
    true_ratio = history["true_ratio_mean"][idx]
    q_ratio = history["q_ratio_mean"][idx]
    late_ratio = history["ratio_mean"][late_mask]
    late_true_ratio = history["true_ratio_mean"][late_mask]
    log_ratio_error = np.abs(
        np.log(np.maximum(ratio, 1e-12)) - np.log(np.maximum(true_ratio, 1e-12))
    )
    late_log_ratio_error = np.abs(
        np.log(np.maximum(late_ratio, 1e-12)) - np.log(np.maximum(late_true_ratio, 1e-12))
    )
    log_q_ratio_error = np.abs(
        np.log(np.maximum(q_ratio, 1e-12)) - np.log(np.maximum(ratio, 1e-12))
    )
    late_q_mu_log_error = np.abs(
        np.log(np.maximum(history["q_mu_mean"][late_mask], 1e-12))
        - np.log(np.maximum(history["true_q_mu_mean"][late_mask], 1e-12))
    )

    summary = {
        "name": config.name,
        "mode": config.mode,
        "config": asdict(config),
        "sfec_true_mean_mse": float(history["sfec_true_mse"][idx].mean()),
        "mu_star_true_mean_mse": float(history["mu_star_true_mse"][idx].mean()),
        "kalman_true_mean_mse": float(history["kalman_true_mse"][idx].mean()),
        "sfec_mu_star_mean_mse": float(history["sfec_mu_star_mse"][idx].mean()),
        "sfec_kalman_mean_mse": float(history["sfec_kalman_mse"][idx].mean()),
        "mu_star_kalman_mean_mse": float(history["mu_star_kalman_mse"][idx].mean()),
        "sfec_true_final_mse": float(history["sfec_true_mse"][-1]),
        "sfec_mu_star_final_mse": float(history["sfec_mu_star_mse"][-1]),
        "mu_star_kalman_final_mse": float(history["mu_star_kalman_mse"][-1]),
        "mean_spikes_per_step": float(history["spike_count"][idx].mean()),
        "p95_spikes_per_step": float(np.quantile(history["spike_count"][idx], 0.95)),
        "zero_spike_fraction": float(np.mean(history["spike_count"][idx] == 0.0)),
        "mean_active_fraction": float(history["active_fraction"][idx].mean()),
        "mean_free_energy_drop": float(history["free_energy_drop"][idx].mean()),
        "positive_free_energy_drop_fraction": float(np.mean(history["free_energy_drop"][idx] >= -1e-10)),
        "pi_y_end_over_true": float(history["pi_y_mean"][-1] / max(history["true_pi_y_mean"][-1], 1e-12)),
        "pi_mu_end_over_true": float(history["pi_mu_mean"][-1] / max(history["true_pi_mu_mean"][-1], 1e-12)),
        "q_y_end_over_true": float(history["q_y_mean"][-1] / max(history["true_q_y_mean"][-1], 1e-12)),
        "q_mu_end_over_true": float(history["q_mu_mean"][-1] / max(history["true_q_mu_mean"][-1], 1e-12)),
        "q_y_target_end_over_true": float(history["q_y_target_mean"][-1] / max(history["true_q_y_mean"][-1], 1e-12)),
        "q_mu_target_end_over_true": float(history["q_mu_target_mean"][-1] / max(history["true_q_mu_mean"][-1], 1e-12)),
        "k_target_end": float(history["k_target_mean"][-1]),
        "current_gain_end": float(history["current_gain_mean"][-1]),
        "true_gain_end": float(history["true_gain_mean"][-1]),
        "window_activity_end": float(history["window_activity"][-1]),
        "ratio_end": float(history["ratio_mean"][-1]),
        "true_ratio_end": float(history["true_ratio_mean"][-1]),
        "q_ratio_end": float(history["q_ratio_mean"][-1]),
        "ratio_end_over_true": float(history["ratio_mean"][-1] / max(history["true_ratio_mean"][-1], 1e-12)),
        "q_ratio_end_over_true": float(history["q_ratio_mean"][-1] / max(history["true_ratio_mean"][-1], 1e-12)),
        "q_ratio_end_over_ratio": float(history["q_ratio_mean"][-1] / max(history["ratio_mean"][-1], 1e-12)),
        "mean_abs_log_ratio_error": float(np.mean(log_ratio_error)),
        "final_abs_log_ratio_error": float(log_ratio_error[-1]),
        "mean_abs_log_q_ratio_error": float(np.mean(log_q_ratio_error)),
        "final_abs_log_q_ratio_error": float(log_q_ratio_error[-1]),
        "mean_ratio_consistency_error": float(history["ratio_consistency_error"][idx].mean()),
        "late_start_time": float(late_start_time),
        "late_sfec_true_mean_mse": float(history["sfec_true_mse"][late_mask].mean()),
        "late_mu_star_kalman_mean_mse": float(history["mu_star_kalman_mse"][late_mask].mean()),
        "late_mean_spikes_per_step": float(history["spike_count"][late_mask].mean()),
        "late_q_y_mean_over_true": float(
            np.mean(history["q_y_mean"][late_mask]) / max(np.mean(history["true_q_y_mean"][late_mask]), 1e-12)
        ),
        "late_q_mu_mean_over_true": float(
            np.mean(history["q_mu_mean"][late_mask]) / max(np.mean(history["true_q_mu_mean"][late_mask]), 1e-12)
        ),
        "late_q_mu_min_over_true_min": float(
            np.min(history["q_mu_min"][late_mask]) / max(np.min(history["true_q_mu_min"][late_mask]), 1e-12)
        ),
        "late_ratio_mean_over_true": float(
            np.mean(history["ratio_mean"][late_mask]) / max(np.mean(history["true_ratio_mean"][late_mask]), 1e-12)
        ),
        "late_abs_log_ratio_error": float(np.mean(late_log_ratio_error)),
        "late_abs_log_q_mu_error": float(np.mean(late_q_mu_log_error)),
        "late_k_target_over_current": float(
            np.mean(history["k_target_mean"][late_mask]) / max(np.mean(history["current_gain_mean"][late_mask]), 1e-12)
        ),
        "late_k_target_over_true": float(
            np.mean(history["k_target_mean"][late_mask]) / max(np.mean(history["true_gain_mean"][late_mask]), 1e-12)
        ),
        "late_current_over_true_gain": float(
            np.mean(history["current_gain_mean"][late_mask]) / max(np.mean(history["true_gain_mean"][late_mask]), 1e-12)
        ),
        "mean_abs_gain_id_error": float(
            np.mean(
                np.abs(
                    np.log(np.maximum(history["k_target_mean"][idx], 1e-12))
                    - np.log(np.maximum(history["current_gain_mean"][idx], 1e-12))
                )
            )
        ),
        "q_y_floor_hits_final": int(history["q_y_floor_hits"][-1]),
        "q_mu_floor_hits_final": int(history["q_mu_floor_hits"][-1]),
        "nu_y_mean_norm": float(history["nu_y_norm"][idx].mean()),
        "eps_y_mean_norm": float(history["eps_y_norm"][idx].mean()),
        "nu_mu_mean_norm": float(history["nu_mu_norm"][idx].mean()),
        "eps_mu_mean_norm": float(history["eps_mu_norm"][idx].mean()),
        "delta_mu_mean_norm": float(history["delta_mu_norm"][idx].mean()),
        "mean_abs_log_q_mu_target_error": float(
            np.mean(
                np.abs(
                    np.log(np.maximum(history["q_mu_target_mean"][idx], 1e-12))
                    - np.log(np.maximum(history["true_q_mu_mean"][idx], 1e-12))
                )
            )
        ),
        "mean_window_activity": float(history["window_activity"][idx].mean()),
    }
    return summary


def save_timeseries_csv(history, path):
    columns = [
        history["t"],
        history["sfec_true_mse"],
        history["mu_star_true_mse"],
        history["kalman_true_mse"],
        history["sfec_mu_star_mse"],
        history["sfec_kalman_mse"],
        history["mu_star_kalman_mse"],
        history["spike_count"],
        history["active_fraction"],
        history["free_energy_start"],
        history["free_energy_end"],
        history["free_energy_drop"],
        history["nu_y_norm"],
        history["eps_y_norm"],
        history["nu_mu_norm"],
        history["eps_mu_norm"],
        history["delta_mu_norm"],
        history["pi_y_mean"],
        history["pi_mu_mean"],
        history["q_y_mean"],
        history["q_mu_mean"],
        history["q_y_target_mean"],
        history["q_mu_target_mean"],
        history["k_target_mean"],
        history["current_gain_mean"],
        history["true_gain_mean"],
        history["s_target_mean"],
        history["window_activity"],
        history["true_pi_y_mean"],
        history["true_pi_mu_mean"],
        history["true_q_y_mean"],
        history["true_q_mu_mean"],
        history["ratio_mean"],
        history["true_ratio_mean"],
        history["q_ratio_mean"],
        history["ratio_consistency_error"],
    ]
    headers = [
        "t",
        "sfec_true_mse",
        "mu_star_true_mse",
        "kalman_true_mse",
        "sfec_mu_star_mse",
        "sfec_kalman_mse",
        "mu_star_kalman_mse",
        "spike_count",
        "active_fraction",
        "free_energy_start",
        "free_energy_end",
        "free_energy_drop",
        "nu_y_norm",
        "eps_y_norm",
        "nu_mu_norm",
        "eps_mu_norm",
        "delta_mu_norm",
        "pi_y_mean",
        "pi_mu_mean",
        "q_y_mean",
        "q_mu_mean",
        "q_y_target_mean",
        "q_mu_target_mean",
        "k_target_mean",
        "current_gain_mean",
        "true_gain_mean",
        "s_target_mean",
        "window_activity",
        "true_pi_y_mean",
        "true_pi_mu_mean",
        "true_q_y_mean",
        "true_q_mu_mean",
        "ratio_mean",
        "true_ratio_mean",
        "q_ratio_mean",
        "ratio_consistency_error",
    ]

    for name in ["x", "sfec_mu", "sfec_mu_star", "kalman_mu"]:
        for i in range(history[name].shape[1]):
            columns.append(history[name][:, i])
            headers.append(f"{name}_{i}")

    data = np.column_stack(columns)
    np.savetxt(path, data, delimiter=",", header=",".join(headers), comments="")


def plot_experiment(history, summary, config):
    t = history["t"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(t, history["sfec_true_mse"], label="FESEPL vs true", linewidth=2)
    ax.plot(t, history["sfec_mu_star_mse"], label="FESEPL vs mu_star", linewidth=2)
    ax.plot(t, history["mu_star_kalman_mse"], label="mu_star vs Kalman", linewidth=2)
    ax.plot(t, history["kalman_true_mse"], label="Kalman vs true", linewidth=2, linestyle="--")
    ax.set_yscale("log")
    ax.set_title(f"{config.name} MSE")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for dim in range(history["x"].shape[1]):
        color = colors[dim % len(colors)]
        ax.plot(t, history["x"][:, dim], color=color, linewidth=2, label=f"x[{dim}]")
        ax.plot(t, history["sfec_mu"][:, dim], color=color, linestyle="--", linewidth=2, label=f"FESEPL[{dim}]")
        ax.plot(t, history["sfec_mu_star"][:, dim], color=color, linestyle=":", linewidth=2, label=f"mu_star[{dim}]")
        ax.plot(t, history["kalman_mu"][:, dim], color=color, linestyle="-.", linewidth=2, label=f"kalman[{dim}]")
    ax.set_title("State Trajectories")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1, 0]
    ax.plot(t, history["ratio_mean"], label="learned ratio pi_mu / pi_y", linewidth=2)
    ax.plot(t, history["true_ratio_mean"], label="true ratio", linewidth=2, linestyle="--")
    ax.plot(t, history["q_ratio_mean"], label="trace ratio q_y / q_mu", linewidth=2, linestyle=":")
    ax.set_yscale("log")
    ax.set_title("Precision Ratio")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(t, history["q_y_mean"], label="learned q_y", linewidth=2)
    ax.plot(t, history["q_mu_mean"], label="learned q_mu", linewidth=2)
    ax.plot(t, history["q_mu_target_mean"], label="q_mu target", linewidth=1.5, linestyle=":")
    ax.plot(t, history["true_q_y_mean"], label="true q_y", linewidth=2, linestyle="--")
    ax.plot(t, history["true_q_mu_mean"], label="true q_mu", linewidth=2, linestyle="--")
    ax.set_yscale("log")
    ax.set_title("Channel Variances")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[2, 0]
    ax.plot(t, history["spike_count"], label="spikes", linewidth=2)
    ax.plot(t, history["active_fraction"] * history["x"].shape[1], label="active fraction scaled", linewidth=2)
    ax.plot(t, history["free_energy_drop"], label="F start - F end", linewidth=2)
    ax.set_title("Spiking And Free-Energy Drop")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[2, 1]
    ax.plot(t, history["nu_y_norm"], label="||nu_y||", linewidth=2)
    ax.plot(t, history["eps_y_norm"], label="||eps_y||", linewidth=2)
    ax.plot(t, history["nu_mu_norm"], label="||nu_mu||", linewidth=2)
    ax.plot(t, history["delta_mu_norm"], label="||delta_mu||", linewidth=2)
    ax.plot(t, history["ratio_consistency_error"], label="|pi-ratio - q-ratio|", linewidth=2)
    ax.set_title("Error Snapshot Norms")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(
        f"{config.name}: FESEPL-vs-mu_star={summary['sfec_mu_star_mean_mse']:.3g}, "
        f"ratio/true={summary['ratio_end_over_true']:.3g}, "
        f"spikes/step={summary['mean_spikes_per_step']:.3g}",
        fontsize=12,
    )
    return fig


def plot_sweeps(obs_rows, proc_rows):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax = axes[0]
    obs_noise = [row["observation_noise"] for row in obs_rows]
    learned_ratio = [row["ratio_end"] for row in obs_rows]
    true_ratio = [row["true_ratio_end"] for row in obs_rows]
    ax.plot(obs_noise, learned_ratio, marker="o", linewidth=2, label="learned ratio")
    ax.plot(obs_noise, true_ratio, marker="s", linewidth=2, linestyle="--", label="true ratio")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Observation Noise Sweep")
    ax.set_xlabel("observation noise")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    proc_noise = [row["process_noise"] for row in proc_rows]
    learned_ratio = [row["ratio_end"] for row in proc_rows]
    true_ratio = [row["true_ratio_end"] for row in proc_rows]
    ax.plot(proc_noise, learned_ratio, marker="o", linewidth=2, label="learned ratio")
    ax.plot(proc_noise, true_ratio, marker="s", linewidth=2, linestyle="--", label="true ratio")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Process-Noise Sweep")
    ax.set_xlabel("process noise")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    return fig


def run_observation_noise_sweep(base_config):
    rows = []
    for obs_noise in [0.1, 1.0, 10.0]:
        config = replace(base_config, name=f"obs_noise_{obs_noise:g}", observation_noise=obs_noise)
        _, _, _, history = run_experiment(config)
        summary = summarize_experiment(history, config)
        rows.append(
            {
                "observation_noise": obs_noise,
                "sfec_mu_star_mean_mse": summary["sfec_mu_star_mean_mse"],
                "mu_star_kalman_mean_mse": summary["mu_star_kalman_mean_mse"],
                "ratio_end": summary["ratio_end"],
                "true_ratio_end": summary["true_ratio_end"],
                "ratio_end_over_true": summary["ratio_end_over_true"],
                "late_ratio_mean_over_true": summary["late_ratio_mean_over_true"],
                "q_y_end_over_true": summary["q_y_end_over_true"],
                "q_mu_end_over_true": summary["q_mu_end_over_true"],
                "late_q_mu_mean_over_true": summary["late_q_mu_mean_over_true"],
                "late_q_mu_min_over_true_min": summary["late_q_mu_min_over_true_min"],
                "late_abs_log_ratio_error": summary["late_abs_log_ratio_error"],
                "late_k_target_over_current": summary["late_k_target_over_current"],
                "late_current_over_true_gain": summary["late_current_over_true_gain"],
                "q_mu_floor_hits_final": summary["q_mu_floor_hits_final"],
                "q_ratio_end_over_ratio": summary["q_ratio_end_over_ratio"],
                "mean_spikes_per_step": summary["mean_spikes_per_step"],
            }
        )
    return rows


def run_process_noise_sweep(base_config):
    rows = []
    for proc_noise in [0.005, 0.1, 2.0]:
        config = replace(
            base_config,
            name=f"proc_noise_{proc_noise:g}",
            plant_process_noise=proc_noise,
            internal_process_noise=proc_noise,
        )
        _, _, _, history = run_experiment(config)
        summary = summarize_experiment(history, config)
        rows.append(
            {
                "process_noise": proc_noise,
                "sfec_mu_star_mean_mse": summary["sfec_mu_star_mean_mse"],
                "mu_star_kalman_mean_mse": summary["mu_star_kalman_mean_mse"],
                "ratio_end": summary["ratio_end"],
                "true_ratio_end": summary["true_ratio_end"],
                "ratio_end_over_true": summary["ratio_end_over_true"],
                "late_ratio_mean_over_true": summary["late_ratio_mean_over_true"],
                "q_y_end_over_true": summary["q_y_end_over_true"],
                "q_mu_end_over_true": summary["q_mu_end_over_true"],
                "late_q_mu_mean_over_true": summary["late_q_mu_mean_over_true"],
                "late_q_mu_min_over_true_min": summary["late_q_mu_min_over_true_min"],
                "late_abs_log_ratio_error": summary["late_abs_log_ratio_error"],
                "late_k_target_over_current": summary["late_k_target_over_current"],
                "late_current_over_true_gain": summary["late_current_over_true_gain"],
                "q_mu_floor_hits_final": summary["q_mu_floor_hits_final"],
                "q_ratio_end_over_ratio": summary["q_ratio_end_over_ratio"],
                "mean_spikes_per_step": summary["mean_spikes_per_step"],
            }
        )
    return rows


def rows_to_csv(rows, path):
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    data = np.asarray([[row[h] for h in headers] for row in rows], dtype=object)
    lines = [",".join(headers)]
    for row in data:
        lines.append(",".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(results):
    frozen = results["frozen_true_precisions"]["summary"]
    base = results["base_long_run"]["summary"]
    obs = results["observation_noise_sweep"]
    proc = results["process_noise_sweep"]

    lines = [
        "# State-Space FESEPL Innovation-Whitening Validation",
        "",
        "## Algorithm",
        "",
        "The current slow learner is parameterized by an effective gain `K` and",
        "an innovation scale `S`, not by direct posterior residual identities.",
        "",
        "- keep the exact dynamic membrane fast loop unchanged",
        "- compute the local innovation `nu = y - mu^-` after each outer step",
        "- maintain local traces:",
        "  `z` delayed innovation, `c0` innovation power, `c1` lag-one covariance",
        "- drive the gain state from lag-one innovation whitening:",
        "  positive lag-one covariance means the gain is too small,",
        "  negative lag-one covariance means it is too large",
        "- learn the innovation scale directly from `c0`",
        "- reconstruct the two variances used by the free-energy objective as",
        "  `q_mu = K S` and `q_y = (1 - K) S`",
        "- map those back to channel precisions with hard bounds on `K` and",
        "  conservative step caps on both gain and scale",
        "",
        "The main validation criterion is long-run behaviour rather than short transients:",
        "",
        "- base recovery run: `30 s`",
        "- noise sweeps: `30 s` each",
        "- late-time metrics are computed on the last `10 s` of each run when possible",
        "",
        "## Tuned Setting",
        "",
        "- `tau_q = 0.1`",
        "- `tau_lambda = 0.25`",
        "- `eta_lambda = 0.5`",
        "- `kappa_y = 0.0`",
        "- `kappa_mu = 0.0`",
        "- `gain_min = 0.02`, `gain_max = 0.98`",
        "- `lambda_y_step_max = 0.10`",
        "- `lambda_mu_step_max = 0.10`",
        "- `observation_window_factor = 2.0`",
        "- `prior_window_factor = 1.0`",
        "- `N = 32`, `beta = 0.0`, `n_inner = 80`, `max_spike_rounds = 32`",
        "",
        "The slow learner keeps the two channel-space variances used by the free-energy objective:",
        "",
        "`q_y ~ R`  and  `q_mu ~ P^-`",
        "",
        "and the precision ratio is then a derived quantity",
        "",
        "`rho = pi_mu / pi_y = q_y / q_mu`.",
        "",
        "## What The Gain Diagnostic Says",
        "",
        "- The fast loop itself is still healthy. The frozen-precision run keeps `FESEPL vs mu_star` tiny, so the spiking approximation is not the bottleneck.",
        "- The whitening learner is evaluated by whether the local gain update moves the filter toward the true Kalman gain and whether the resulting `q_y` and `q_mu` stay near their injected long-run values.",
        f"- In the base 30 s run, late `K_target / K_current = {base['late_k_target_over_current']:.6g}` while late `K_current / K_true = {base['late_current_over_true_gain']:.6g}`.",
        "- If `K_target / K_current` stays close to one while `K_current / K_true` stays far from one, the whitening signal is not correcting the gain strongly enough.",
        "",
        "## Frozen True Precisions",
        f"- FESEPL vs true mean MSE: `{frozen['sfec_true_mean_mse']:.6g}`",
        f"- FESEPL vs mu_star mean MSE: `{frozen['sfec_mu_star_mean_mse']:.6g}`",
        f"- mu_star vs Kalman mean MSE: `{frozen['mu_star_kalman_mean_mse']:.6g}`",
        f"- Final learned ratio / true ratio: `{frozen['ratio_end_over_true']:.6g}`",
        f"- Mean spikes per step: `{frozen['mean_spikes_per_step']:.6g}`",
        f"- Positive free-energy-drop fraction: `{frozen['positive_free_energy_drop_fraction']:.6g}`",
        "",
        "## Base Long Run",
        "",
        f"- Final q_y / true R: `{base['q_y_end_over_true']:.6g}`",
        f"- Final q_mu / true P^-: `{base['q_mu_end_over_true']:.6g}`",
        f"- Final ratio / true ratio: `{base['ratio_end_over_true']:.6g}`",
        f"- Final K_target / true gain: `{base['k_target_end'] / max(base['true_gain_end'], 1e-12):.6g}`",
        f"- Final current gain / true gain: `{base['current_gain_end'] / max(base['true_gain_end'], 1e-12):.6g}`",
        f"- Late-window mean q_mu / true P^-: `{base['late_q_mu_mean_over_true']:.6g}`",
        f"- Late-window mean q_y / true R: `{base['late_q_y_mean_over_true']:.6g}`",
        f"- Late-window minimum q_mu / true P^-: `{base['late_q_mu_min_over_true_min']:.6g}`",
        f"- Late-window mean ratio / true ratio: `{base['late_ratio_mean_over_true']:.6g}`",
        f"- Late-window mean abs log ratio error: `{base['late_abs_log_ratio_error']:.6g}`",
        f"- Late-window mean abs log q_mu error: `{base['late_abs_log_q_mu_error']:.6g}`",
        f"- Late-window K_target / current gain: `{base['late_k_target_over_current']:.6g}`",
        f"- Late-window current gain / true gain: `{base['late_current_over_true_gain']:.6g}`",
        f"- Late mu_star vs Kalman MSE: `{base['late_mu_star_kalman_mean_mse']:.6g}`",
        f"- q_mu floor hits: `{base['q_mu_floor_hits_final']}`",
        f"- q_y floor hits: `{base['q_y_floor_hits_final']}`",
        "",
        "## Observation Noise Sweep",
    ]
    for row in obs:
        lines.append(
            f"- noise `{row['observation_noise']}`: final ratio/true `{row['ratio_end_over_true']:.6g}`, "
            f"late ratio/true `{row['late_ratio_mean_over_true']:.6g}`, "
            f"late q_mu/true `{row['late_q_mu_mean_over_true']:.6g}`, "
            f"late q_y/true `{row['q_y_end_over_true']:.6g}`, "
            f"late K_target/current `{row['late_k_target_over_current']:.6g}`, "
            f"late current/true gain `{row['late_current_over_true_gain']:.6g}`, "
            f"late min q_mu/true `{row['late_q_mu_min_over_true_min']:.6g}`, "
            f"late abs log ratio error `{row['late_abs_log_ratio_error']:.6g}`, "
            f"q_mu floor hits `{int(row['q_mu_floor_hits_final'])}`"
        )

    lines.extend(["", "## Process-Noise Sweep"])
    for row in proc:
        lines.append(
            f"- process noise `{row['process_noise']}`: final ratio/true `{row['ratio_end_over_true']:.6g}`, "
            f"late ratio/true `{row['late_ratio_mean_over_true']:.6g}`, "
            f"late q_mu/true `{row['late_q_mu_mean_over_true']:.6g}`, "
            f"late q_y/true `{row['q_y_end_over_true']:.6g}`, "
            f"late K_target/current `{row['late_k_target_over_current']:.6g}`, "
            f"late current/true gain `{row['late_current_over_true_gain']:.6g}`, "
            f"late min q_mu/true `{row['late_q_mu_min_over_true_min']:.6g}`, "
            f"late abs log ratio error `{row['late_abs_log_ratio_error']:.6g}`, "
            f"q_mu floor hits `{int(row['q_mu_floor_hits_final'])}`"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "- The frozen-precision run isolates the fast loop. If `FESEPL vs mu_star` stays tiny there, the spiking approximation is not the main bottleneck.",
            "- The real pass/fail signal is whether long-run innovation whitening drives the gain toward the true one and keeps both `q_y` and `q_mu` near the injected scale.",
            "- `K_target / K_current` close to one with `K_current / K_true` still far from one means the innovation-covariance drive is too weak or too biased to correct the current filter.",
            "- This learner is honest and local, but it still has to be judged by the long-run ratios, not by short transients or by how tidy the equations look.",
        ]
    )

    return "\n".join(lines) + "\n"


def main():
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_config = RunConfig(
        name="frozen_true_precisions",
        mode="freeze_true",
        total_time=2.0,
        plant_process_noise=0.02,
        internal_process_noise=0.02,
        N=32,
        beta=0.0,
        n_inner=80,
        max_spike_rounds=32,
        tau_q=0.1,
        tau_lambda=0.25,
        eta_lambda=0.5,
        kappa_y=0.0,
        kappa_mu=0.0,
        lambda_y_step_max=0.10,
        lambda_mu_step_max=0.10,
        observation_window_factor=2.0,
        prior_window_factor=1.0,
    )
    base_learned = RunConfig(
        name="base_long_run",
        mode="learn",
        total_time=30.0,
        plant_process_noise=0.02,
        internal_process_noise=0.02,
        N=32,
        beta=0.0,
        n_inner=80,
        max_spike_rounds=32,
        tau_q=0.1,
        tau_lambda=0.25,
        eta_lambda=0.5,
        kappa_y=0.0,
        kappa_mu=0.0,
        observation_variance_guess=0.5,
        prior_variance_guess=5.0,
        lambda_y_step_max=0.10,
        lambda_mu_step_max=0.10,
        observation_window_factor=2.0,
        prior_window_factor=1.0,
    )
    sweep_template = replace(base_learned, total_time=30.0)

    results = {}

    for config in [frozen_config]:
        _, estimator, _, history = run_experiment(config)
        summary = summarize_experiment(history, config)
        fig = plot_experiment(history, summary, config)
        fig_path = output_dir / f"{config.name}.png"
        csv_path = output_dir / f"{config.name}_timeseries.csv"
        json_path = output_dir / f"{config.name}_summary.json"
        npz_path = output_dir / f"{config.name}.npz"

        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        save_timeseries_csv(history, csv_path)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        np.savez(npz_path, **history)

        results[config.name] = {
            "summary": summary,
            "figure": str(fig_path),
            "csv": str(csv_path),
            "json": str(json_path),
            "npz": str(npz_path),
            "final_spikes": float(estimator.spike_totals.sum()),
        }

    _, estimator, _, history = run_experiment(base_learned)
    summary = summarize_experiment(history, base_learned)
    fig = plot_experiment(history, summary, base_learned)
    fig_path = output_dir / f"{base_learned.name}.png"
    csv_path = output_dir / f"{base_learned.name}_timeseries.csv"
    json_path = output_dir / f"{base_learned.name}_summary.json"
    npz_path = output_dir / f"{base_learned.name}.npz"

    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    save_timeseries_csv(history, csv_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(npz_path, **history)

    results[base_learned.name] = {
        "summary": summary,
        "figure": str(fig_path),
        "csv": str(csv_path),
        "json": str(json_path),
        "npz": str(npz_path),
        "final_spikes": float(estimator.spike_totals.sum()),
    }

    obs_rows = run_observation_noise_sweep(sweep_template)
    proc_rows = run_process_noise_sweep(sweep_template)
    sweep_fig = plot_sweeps(obs_rows, proc_rows)
    sweep_fig_path = output_dir / "sweeps.png"
    sweep_fig.savefig(sweep_fig_path, dpi=180)
    plt.close(sweep_fig)

    rows_to_csv(obs_rows, output_dir / "observation_noise_sweep.csv")
    rows_to_csv(proc_rows, output_dir / "process_noise_sweep.csv")

    results["observation_noise_sweep"] = obs_rows
    results["process_noise_sweep"] = proc_rows
    results["sweeps_figure"] = str(sweep_fig_path)

    report = build_report(results)
    report_path = output_dir / "state_space_report.md"
    json_path = output_dir / "state_space_experiments.json"

    report_path.write_text(report, encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(report)
    print("Saved Outputs:")
    for key, value in results.items():
        if isinstance(value, dict):
            for sub_key in ["figure", "csv", "json", "npz"]:
                if sub_key in value:
                    print(f"  {key}_{sub_key}: {value[sub_key]}")
    print(f"  report: {report_path}")
    print(f"  summary_json: {json_path}")


if __name__ == "__main__":
    main()
