import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MPL_DIR = ROOT / "artifacts" / "mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_FESEPL import (
    RunConfig,
    build_estimator,
    build_kalman,
    build_plant,
    compute_true_observation_channel_stats,
    compute_true_prior_channel_stats,
    frozen_true_precision_step,
    kalman_correct_from_prior,
    make_initial_state_guess,
    summarize_experiment,
)


@dataclass
class RunnerConfig(RunConfig):
    name: str = "state_space_single_run"
    mode: str = "learn"
    total_time: float = 30.0

    observation_noise: float = 0.1
    plant_process_noise: float = 0.5
    internal_process_noise: float | None = None
    observation_variance_guess: float = 0.5
    prior_variance_guess: float = 5.0

    N: int = 32
    beta: float = 0.0
    tau_q: float = 0.1
    tau_lambda: float = 0.25
    eta_lambda: float = 0.5
    kappa_y: float = 0.0
    kappa_mu: float = 0.0
    lambda_y_step_max: float = 0.10
    lambda_mu_step_max: float = 0.10
    observation_window_factor: float = 2.0
    prior_window_factor: float = 1.0
    n_inner: int = 80
    max_spike_rounds: int = 32

    plot_state_dims: int = 2
    raster_neurons: int = 32
    save_dir: str = "runner_outputs"


def allocate_history(steps, plant, estimator, config):
    state_dims = min(config.plot_state_dims, plant.x_k)
    raster_neurons = min(config.raster_neurons, estimator.N)

    history = {
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
        "pi_y_channels": np.zeros((steps + 1, estimator.m)),
        "pi_mu_channels": np.zeros((steps + 1, estimator.n)),
        "q_y_channels": np.zeros((steps + 1, estimator.m)),
        "q_mu_channels": np.zeros((steps + 1, estimator.n)),
        "true_pi_y_channels": np.zeros((steps + 1, estimator.m)),
        "true_pi_mu_channels": np.zeros((steps + 1, estimator.n)),
        "true_q_y_channels": np.zeros((steps + 1, estimator.m)),
        "true_q_mu_channels": np.zeros((steps + 1, estimator.n)),
        "spike_raster": np.zeros((steps, raster_neurons)),
        "state_dims": state_dims,
        "raster_neurons": raster_neurons,
    }
    return history


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

    history["pi_y_channels"][idx] = estimator.pi_y
    history["pi_mu_channels"][idx] = estimator.pi_mu
    history["q_y_channels"][idx] = estimator.q_y
    history["q_mu_channels"][idx] = estimator.q_mu
    history["true_pi_y_channels"][idx] = true_pi_y
    history["true_pi_mu_channels"][idx] = true_pi_mu
    history["true_q_y_channels"][idx] = true_q_y
    history["true_q_mu_channels"][idx] = true_q_mu


def run_single(config):
    plant = build_plant(config)
    estimator = build_estimator(plant, config)
    kalman = build_kalman(plant, config)

    steps = int(config.total_time / config.dt)
    history = allocate_history(steps, plant, estimator, config)

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

        history["spike_raster"][k] = estimator.spike_totals[: history["raster_neurons"]]

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
        history["q_y_min"],
        history["q_mu_min"],
        history["true_q_y_min"],
        history["true_q_mu_min"],
        history["q_y_floor_hits"],
        history["q_mu_floor_hits"],
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
        "q_y_min",
        "q_mu_min",
        "true_q_y_min",
        "true_q_mu_min",
        "q_y_floor_hits",
        "q_mu_floor_hits",
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

    for name in [
        "pi_y_channels",
        "pi_mu_channels",
        "q_y_channels",
        "q_mu_channels",
        "true_pi_y_channels",
        "true_pi_mu_channels",
        "true_q_y_channels",
        "true_q_mu_channels",
    ]:
        for i in range(history[name].shape[1]):
            columns.append(history[name][:, i])
            headers.append(f"{name}_{i}")

    data = np.column_stack(columns)
    np.savetxt(path, data, delimiter=",", header=",".join(headers), comments="")


def format_summary(summary):
    cfg = summary["config"]
    internal_process_noise = cfg["plant_process_noise"] if cfg["internal_process_noise"] is None else cfg["internal_process_noise"]
    mismatch_note = ""
    if abs(internal_process_noise - cfg["plant_process_noise"]) > 1e-15:
        mismatch_note = " [model mismatch]"
    lines = [
        "State-Space Runner",
        "",
        "Noise Setup:",
        f"  plant process noise: {cfg['plant_process_noise']:.6g}",
        f"  internal process noise: {internal_process_noise:.6g}{mismatch_note}",
        f"  observation noise: {cfg['observation_noise']:.6g}",
        "",
        "MSE:",
        f"  FESEPL vs true mean/final: {summary['sfec_true_mean_mse']:.6g} / {summary['sfec_true_final_mse']:.6g}",
        f"  FESEPL vs mu_star mean/final: {summary['sfec_mu_star_mean_mse']:.6g} / {summary['sfec_mu_star_final_mse']:.6g}",
        f"  mu_star vs Kalman mean/final: {summary['mu_star_kalman_mean_mse']:.6g} / {summary['mu_star_kalman_final_mse']:.6g}",
        "",
        "Spikes:",
        f"  mean/p95 spikes per step: {summary['mean_spikes_per_step']:.6g} / {summary['p95_spikes_per_step']:.6g}",
        f"  zero-spike fraction: {summary['zero_spike_fraction']:.6g}",
        f"  mean active fraction: {summary['mean_active_fraction']:.6g}",
        "",
        "Precisions:",
        f"  final pi_y / true: {summary['pi_y_end_over_true']:.6g}",
        f"  final pi_mu / true: {summary['pi_mu_end_over_true']:.6g}",
        f"  final q_y / true: {summary['q_y_end_over_true']:.6g}",
        f"  final q_mu / true: {summary['q_mu_end_over_true']:.6g}",
        f"  late q_y / true after {summary['late_start_time']:.1f}s: {summary['late_q_y_mean_over_true']:.6g}",
        f"  late q_mu / true after {summary['late_start_time']:.1f}s: {summary['late_q_mu_mean_over_true']:.6g}",
        f"  late min q_mu / true after {summary['late_start_time']:.1f}s: {summary['late_q_mu_min_over_true_min']:.6g}",
        f"  late ratio / true after {summary['late_start_time']:.1f}s: {summary['late_ratio_mean_over_true']:.6g}",
        f"  late abs log ratio error: {summary['late_abs_log_ratio_error']:.6g}",
        f"  q_mu floor hits: {summary['q_mu_floor_hits_final']}",
        f"  q_y floor hits: {summary['q_y_floor_hits_final']}",
        "",
        "Free Energy:",
        f"  mean F drop: {summary['mean_free_energy_drop']:.6g}",
        f"  positive F-drop fraction: {summary['positive_free_energy_drop_fraction']:.6g}",
        "",
        "Error Norms:",
        f"  mean ||nu_y|| / ||eps_y||: {summary['nu_y_mean_norm']:.6g} / {summary['eps_y_mean_norm']:.6g}",
        f"  mean ||nu_mu|| / ||eps_mu|| / ||delta_mu||: {summary['nu_mu_mean_norm']:.6g} / {summary['eps_mu_mean_norm']:.6g} / {summary['delta_mu_mean_norm']:.6g}",
        "",
    ]
    return "\n".join(lines)


def plot_runner(history, summary, config):
    t = history["t"]
    t_spike = t[1:]
    q_y_target_vis = history["q_y_target_mean"].copy()
    q_y_target_vis[q_y_target_vis <= 1e-7] = np.nan
    q_mu_target_vis = history["q_mu_target_mean"].copy()
    q_mu_target_vis[q_mu_target_vis <= 1e-7] = np.nan
    fig, axes = plt.subplots(5, 2, figsize=(17, 18), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(t, history["sfec_true_mse"], label="FESEPL vs true", linewidth=2)
    ax.plot(t, history["sfec_mu_star_mse"], label="FESEPL vs mu_star", linewidth=1.8)
    ax.plot(t, history["mu_star_kalman_mse"], label="mu_star vs Kalman", linewidth=1.8)
    ax.plot(t, history["kalman_true_mse"], label="Kalman vs true", linewidth=1.5, linestyle="--")
    ax.set_yscale("log")
    ax.set_title("MSE")
    ax.axvline(summary["late_start_time"], color="k", linestyle=":", linewidth=1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for dim in range(history["state_dims"]):
        color = colors[dim % len(colors)]
        ax.plot(t, history["x"][:, dim], color=color, linewidth=2, label=f"x[{dim}]")
        ax.plot(t, history["sfec_mu"][:, dim], color=color, linestyle="--", linewidth=2, label=f"FESEPL[{dim}]")
        ax.plot(t, history["sfec_mu_star"][:, dim], color=color, linestyle=":", linewidth=2, label=f"mu_star[{dim}]")
        ax.plot(t, history["kalman_mu"][:, dim], color=color, linestyle="-.", linewidth=2, label=f"kalman[{dim}]")
    ax.set_title("State Tracks")
    ax.axvline(summary["late_start_time"], color="k", linestyle=":", linewidth=1)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1, 0]
    for i in range(history["pi_y_channels"].shape[1]):
        ax.plot(
            t,
            history["pi_y_channels"][:, i],
            linewidth=2,
            label=f"pi_y[{i}]",
            drawstyle="steps-post",
        )
        ax.plot(t, history["true_pi_y_channels"][:, i], linewidth=1.5, linestyle="--", label=f"true pi_y[{i}]")
    ax.set_yscale("log")
    ax.set_title("Observation Precisions")
    ax.axvline(summary["late_start_time"], color="k", linestyle=":", linewidth=1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for i in range(history["pi_mu_channels"].shape[1]):
        ax.plot(
            t,
            history["pi_mu_channels"][:, i],
            linewidth=2,
            label=f"pi_mu[{i}]",
            drawstyle="steps-post",
        )
        ax.plot(t, history["true_pi_mu_channels"][:, i], linewidth=1.5, linestyle="--", label=f"true pi_mu[{i}]")
    ax.set_yscale("log")
    ax.set_title("Prior Precisions")
    ax.axvline(summary["late_start_time"], color="k", linestyle=":", linewidth=1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    for i in range(history["q_y_channels"].shape[1]):
        ax.plot(
            t,
            history["q_y_channels"][:, i],
            linewidth=2,
            label=f"q_y[{i}]",
            drawstyle="steps-post",
        )
        ax.plot(t, history["true_q_y_channels"][:, i], linewidth=1.5, linestyle="--", label=f"true q_y[{i}]")
    ax.plot(
        t,
        q_y_target_vis,
        color="k",
        linewidth=1.5,
        linestyle=":",
        label="q_y target mean",
        drawstyle="steps-post",
    )
    ax.set_yscale("log")
    ax.set_title("Observation Variances")
    ax.axvline(summary["late_start_time"], color="k", linestyle=":", linewidth=1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    for i in range(history["q_mu_channels"].shape[1]):
        ax.plot(
            t,
        history["q_mu_channels"][:, i],
        linewidth=2,
        label=f"q_mu[{i}]",
        drawstyle="steps-post",
        )
        ax.plot(t, history["true_q_mu_channels"][:, i], linewidth=1.5, linestyle="--", label=f"true q_mu[{i}]")
    ax.plot(
        t,
        q_mu_target_vis,
        color="k",
        linewidth=1.5,
        linestyle=":",
        label="q_mu target",
        drawstyle="steps-post",
    )
    ax.set_yscale("log")
    ax.set_title("Prior Variances")
    ax.axvline(summary["late_start_time"], color="k", linestyle=":", linewidth=1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[3, 0]
    ax.plot(t, history["ratio_mean"], linewidth=2, label="pi_mu / pi_y", drawstyle="steps-post")
    ax.plot(t, history["q_ratio_mean"], linewidth=1.8, linestyle="--", label="q_y / q_mu", drawstyle="steps-post")
    ax.plot(t, history["true_ratio_mean"], linewidth=1.8, linestyle=":", label="true ratio")
    ax.set_yscale("log")
    ax.set_title("Precision Ratio")
    ax.axvline(summary["late_start_time"], color="k", linestyle=":", linewidth=1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[3, 1]
    qy_norm = history["q_y_mean"] / np.maximum(history["true_q_y_mean"], 1e-12)
    qmu_norm = history["q_mu_mean"] / np.maximum(history["true_q_mu_mean"], 1e-12)
    ax.plot(t, qy_norm, linewidth=2, label="q_y / true R", drawstyle="steps-post")
    ax.plot(t, qmu_norm, linewidth=2, label="q_mu / true P^-", drawstyle="steps-post")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_title("Normalized Variances")
    ax.axvline(summary["late_start_time"], color="k", linestyle=":", linewidth=1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[4, 0]
    ax.plot(t, history["spike_count"], linewidth=1.8, label="spikes per step", color="tab:blue")
    ax.plot(
        t,
        history["active_fraction"] * history["raster_neurons"],
        linewidth=1.5,
        label="active fraction scaled",
        color="tab:orange",
    )
    ax.set_title("Spike Activity And Free-Energy Drop")
    ax.axvline(summary["late_start_time"], color="k", linestyle=":", linewidth=1)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, history["free_energy_drop"], linewidth=1.5, color="tab:green", label="F start - F end")
    ax.set_ylabel("spikes / active units")
    ax2.set_ylabel("free-energy drop")
    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines if not line.get_label().startswith("_")]
    lines = [line for line in lines if not line.get_label().startswith("_")]
    ax.legend(lines, labels, fontsize=8, loc="upper right")

    ax = axes[4, 1]
    raster = history["spike_raster"].T
    positive = raster[raster > 0.0]
    if positive.size:
        vmax = max(1.0, float(np.quantile(positive, 0.95)))
    else:
        vmax = 1.0
    im = ax.imshow(
        raster,
        aspect="auto",
        origin="lower",
        extent=[t_spike[0], t_spike[-1], 0, history["raster_neurons"]],
        cmap="Greys",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title("Spike Raster / Heatmap")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("neuron index")
    ax.axvline(summary["late_start_time"], color="tab:red", linestyle=":", linewidth=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="spikes in step")

    fig.suptitle(
        f"{config.name}: FESEPL-vs-mu_star={summary['sfec_mu_star_mean_mse']:.3g}, "
        f"late q_y/true={summary['late_q_y_mean_over_true']:.3g}, "
        f"late q_mu/true={summary['late_q_mu_mean_over_true']:.3g}, "
        f"late ratio/true={summary['late_ratio_mean_over_true']:.3g}",
        fontsize=12,
    )
    return fig


def main():
    config = RunnerConfig()
    output_dir = ROOT / config.save_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, _, history = run_single(config)
    summary = summarize_experiment(history, config)
    report = format_summary(summary)
    fig = plot_runner(history, summary, config)

    fig_path = output_dir / f"{config.name}.png"
    csv_path = output_dir / f"{config.name}_timeseries.csv"
    json_path = output_dir / f"{config.name}_summary.json"
    txt_path = output_dir / f"{config.name}_summary.txt"
    npz_path = output_dir / f"{config.name}.npz"

    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    save_timeseries_csv(history, csv_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt_path.write_text(report, encoding="utf-8")
    np.savez(npz_path, **history)

    print(report)
    print("Saved Outputs:")
    print(f"  figure: {fig_path}")
    print(f"  timeseries_csv: {csv_path}")
    print(f"  summary_json: {json_path}")
    print(f"  summary_txt: {txt_path}")
    print(f"  data_npz: {npz_path}")


if __name__ == "__main__":
    main()
