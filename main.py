"""
main.py
=======
Entry point for the UAV-ISAC optimisation simulation.

Implements the system from:
  Liu et al., "UAV Assisted Integrated Sensing and Communications for IoT:
  3D Trajectory Optimization and Resource Allocation," IEEE TWC 2024.

Runs BOTH:
  Baseline  - original Algorithm 2 (single-objective, no fairness)
  Improved  - multi-objective + min-rate fairness constraint

Outputs:
  output_plots_baseline/   <- all individual paper-style plots for baseline
  output_plots_improved/   <- all individual paper-style plots for improved
  output_plots_comparison/ <- 4 comparison figures:
                               fig1_convergence_comparison.png
                               fig2_power_comparison.png
                               fig3_task_scheduling_comparison.png
                               fig4_trajectory_comparison.png
"""

import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from environment import Environment
from uav import UAV

# ── Baseline imports ──────────────────────────────────────────────────────────
from optimization_baseline import (
    three_layer_optimize,
    compute_total_radar_rate as compute_rate_baseline,
    _channel_gains,
    solve_scheduling,
)

# ── Improved imports ──────────────────────────────────────────────────────────
from optimization_improved import (
    _channel_gains as _channel_gains_imp,
    solve_scheduling_fair,
    solve_power_multiobjective,
    solve_trajectory,
    compute_total_radar_rate as compute_rate_improved,
    compute_energy,
)

from utils import h_rad, sinr_rad, radar_rate, distance_3d

# ─────────────────────────────────────────────────────────────────────────────
# Output directories
# ─────────────────────────────────────────────────────────────────────────────
for d in ["output_plots_baseline", "output_plots_improved", "output_plots_comparison"]:
    os.makedirs(d, exist_ok=True)

R_MIN = 0.5   # fairness floor (bps/Hz per node)
LAM   = 0.8   # Pareto weight for improved

# Colour constants
C_BLUE   = "#1565C0"
C_GREEN  = "#2E7D32"
C_RED    = "#C62828"
C_ORANGE = "darkorange"


# =============================================================================
# Shared drawing helpers  (draw onto an existing Axes object)
# =============================================================================

def _draw_scheduling(ax, omega, b, env, title="UAV Task Scheduling per Time Slot"):
    """Paper Fig.4 style task-scheduling colour bar on ax."""
    Q, K = omega.shape
    sched = np.full(Q, -1, dtype=int)
    for q in range(Q):
        if b[q] > 0.5:
            sched[q] = 0
        else:
            for k in range(K):
                if omega[q, k] > 0.5:
                    sched[q] = k + 1

    cmap = plt.cm.tab20(np.linspace(0, 1, K + 1))
    bar_colors = [cmap[s] if s >= 0 else "grey" for s in sched]
    ax.bar(range(Q), np.ones(Q), color=bar_colors, width=1.0)

    patches = [Patch(color=cmap[0], label="Data Centre")]
    for k in range(K):
        patches.append(Patch(color=cmap[k + 1], label=f"Node {k+1}"))
    ax.legend(handles=patches, fontsize=7, ncol=4, loc="upper right", framealpha=0.8)
    ax.set_xlabel("Time Slot Index", fontsize=11)
    ax.set_ylabel("Assigned Target", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.3)


def _draw_top_view(ax, init_pos, final_pos, env, omega, title="Top-View Trajectory"):
    """Paper Fig.3 style top-view trajectory on ax."""
    K = env.num_nodes
    Q = final_pos.shape[0]

    ax.scatter(env.nodes[:, 0], env.nodes[:, 1],
               c="red", s=60, marker="^", zorder=5, label="IoT Nodes")
    ax.scatter(*env.data_center, c="black", s=100, marker="*",
               zorder=6, label="Data Centre")
    for k in range(K):
        ax.annotate(str(k + 1), env.nodes[k], fontsize=7, color="darkred",
                    xytext=(4, 4), textcoords="offset points")

    ax.plot(init_pos[:, 0], init_pos[:, 1],
            color=C_ORANGE, linewidth=1.2, alpha=0.6, label="Initial")

    if omega is not None:
        for q in range(Q - 1):
            isac_slot = any(omega[q, k] > 0.5 for k in range(K))
            col = C_BLUE if isac_slot else C_GREEN
            ax.plot(final_pos[q:q+2, 0], final_pos[q:q+2, 1],
                    color=col, linewidth=1.6)
        extra = [
            Line2D([0], [0], color=C_BLUE,  lw=2, label="ISAC"),
            Line2D([0], [0], color=C_GREEN, lw=2, label="Upload"),
        ]
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + extra, fontsize=8)
    else:
        ax.plot(final_pos[:, 0], final_pos[:, 1], color=C_BLUE, lw=2, label="Optimised")
        ax.legend(fontsize=8)

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


# =============================================================================
# Individual plot generator  (all paper-style plots for one run)
# =============================================================================

def generate_individual_plots(out_dir, label,
                               init_pos, final_pos,
                               init_vel,  final_vel,
                               uav, env, rates, omega, b):
    """Save all individual paper-style plots into out_dir."""
    Q = uav.Q
    K = env.num_nodes

    def _save(fig, name):
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved -> {path}")

    # ------------------------------------------------------------------
    # Plot 1: 3-D Trajectory  (Fig. 2 of paper)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(env.nodes[:, 0], env.nodes[:, 1], np.zeros(K),
               c="red", s=60, marker="^", zorder=5, label="IoT Nodes")
    ax.scatter(*env.data_center, 0, c="black", s=100, marker="*",
               zorder=6, label="Data Centre")
    for k in range(K):
        ax.text(env.nodes[k, 0], env.nodes[k, 1], 5,
                str(k + 1), fontsize=7, color="darkred")

    ax.plot(init_pos[:, 0], init_pos[:, 1], init_pos[:, 2],
            color=C_ORANGE, lw=1.2, alpha=0.6, label="Initial trajectory")

    if omega is not None:
        for q in range(Q - 1):
            col = C_BLUE if any(omega[q, k] > 0.5 for k in range(K)) else C_GREEN
            ax.plot(final_pos[q:q+2, 0], final_pos[q:q+2, 1], final_pos[q:q+2, 2],
                    color=col, lw=2)
        extra = [Line2D([0],[0], color=C_BLUE,  lw=2, label="ISAC"),
                 Line2D([0],[0], color=C_GREEN, lw=2, label="Upload")]
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + extra, fontsize=8)
    else:
        ax.plot(final_pos[:, 0], final_pos[:, 1], final_pos[:, 2],
                color=C_BLUE, lw=2, label="Optimised")
        ax.legend(fontsize=8)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Altitude (m)")
    ax.set_title(f"[{label}] Initial vs. Optimised 3-D UAV Trajectory", fontsize=12)
    _save(fig, "3d_trajectory.png")

    # ------------------------------------------------------------------
    # Plot 2: Top-view trajectory  (Fig. 3 of paper)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_top_view(ax, init_pos, final_pos, env, omega,
                   title=f"[{label}] Top View – Initial vs. Optimised Trajectory")
    _save(fig, "top_view_trajectory.png")

    # ------------------------------------------------------------------
    # Plot 3: Convergence  (Fig. 9 / 12 of paper)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(rates) + 1), rates,
            color=C_BLUE, lw=2.5, marker="o", ms=6, label=label)
    ax.set_xlabel("Number of Iterations", fontsize=12)
    ax.set_ylabel("Sum Radar Estimation Rate (bps/Hz)", fontsize=12)
    ax.set_title(f"[{label}] Convergence of Three-Layer Optimisation", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
    _save(fig, "convergence_rate.png")

    # ------------------------------------------------------------------
    # Plot 4: Task scheduling  (Fig. 4 of paper)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    _draw_scheduling(ax, omega, b, env,
                     title=f"[{label}] UAV Task Scheduling per Time Slot")
    fig.tight_layout()
    _save(fig, "task_scheduling.png")

    # ------------------------------------------------------------------
    # Plot 5: Altitude profile
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(Q), init_pos[:, 2], color=C_ORANGE, lw=1.5, alpha=0.7, label="Initial")
    ax.plot(range(Q), final_pos[:, 2], color=C_BLUE, lw=2, label="Optimised")
    ax.axhline(uav.H_MIN, color="grey", ls="--", lw=0.8, label="H_min")
    ax.axhline(uav.H_MAX, color="grey", ls=":",  lw=0.8, label="H_max")
    ax.set_xlabel("Time Slot Index", fontsize=12)
    ax.set_ylabel("Altitude (m)", fontsize=12)
    ax.set_title(f"[{label}] UAV Flight Altitude Profile", fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.4)
    _save(fig, "altitude_profile.png")

    # ------------------------------------------------------------------
    # Plot 6: Speed profile  (Fig. 6 of paper)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    i_spd = np.linalg.norm(init_vel,      axis=1)
    f_spd = np.linalg.norm(final_vel,     axis=1)
    h_spd = np.linalg.norm(final_vel[:, :2], axis=1)
    v_spd = np.abs(final_vel[:, 2])
    ax.plot(range(Q), i_spd, color=C_ORANGE, lw=1.2, alpha=0.6, label="Initial speed")
    ax.plot(range(Q), f_spd, color=C_BLUE,   lw=2,              label="Optimised total speed")
    ax.plot(range(Q), h_spd, color=C_GREEN,  lw=1.2, ls="--",  label="Optimised horizontal speed")
    ax.plot(range(Q), v_spd, color=C_RED,    lw=1.2, ls=":",   label="Optimised vertical speed")
    ax.set_xlabel("Time Slot Index", fontsize=12)
    ax.set_ylabel("Speed (m/s)", fontsize=12)
    ax.set_title(f"[{label}] UAV Speed Profile", fontsize=13)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    _save(fig, "speed_profile.png")

    # ------------------------------------------------------------------
    # Plot 7: Power allocation  (Fig. 8 of paper)
    # ------------------------------------------------------------------
    slots = np.arange(Q)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].bar(slots, uav.Pcom, label="P_com (α·Pt)",        color=C_BLUE, width=1.0)
    axes[0].bar(slots, uav.Prad, bottom=uav.Pcom,
                label="P_rad ((1-α)·Pt)", color=C_RED,  width=1.0)
    axes[0].set_ylabel("Power (W)", fontsize=11)
    axes[0].set_title(f"[{label}] Transmit Power Allocation per Time Slot", fontsize=12)
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.4)
    axes[1].plot(slots, uav.alpha, color="#ff7f0e", lw=1.5)
    axes[1].axhline(0.5, color="grey", ls="--", lw=0.8)
    axes[1].set_ylabel("α (comm. fraction)", fontsize=11)
    axes[1].set_xlabel("Time Slot Index", fontsize=11)
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"[{label}] Power Split Factor α", fontsize=12)
    axes[1].grid(True, alpha=0.4)
    fig.tight_layout()
    _save(fig, "power_allocation.png")

    # ------------------------------------------------------------------
    # Plot 8: Dashboard  (6-panel summary)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"UAV-ISAC Optimisation Dashboard — {label}",
                 fontsize=14, fontweight="bold")

    # Panel 1: top-view trajectory
    _draw_top_view(axes[0, 0], init_pos, final_pos, env, omega,
                   title="Top-View Trajectory")

    # Panel 2: convergence
    axes[0, 1].plot(range(1, len(rates) + 1), rates,
                    marker="o", lw=2, color=C_BLUE)
    axes[0, 1].set_title("Convergence")
    axes[0, 1].set_xlabel("Iteration"); axes[0, 1].set_ylabel("Rate (bps/Hz)")
    axes[0, 1].grid(True, alpha=0.4)

    # Panel 3: altitude
    axes[0, 2].plot(range(Q), init_pos[:, 2], color=C_ORANGE, lw=1, alpha=0.6, label="Init")
    axes[0, 2].plot(range(Q), final_pos[:, 2], color=C_BLUE, lw=2, label="Opt")
    axes[0, 2].axhline(uav.H_MIN, color="grey", ls="--", lw=0.8)
    axes[0, 2].axhline(uav.H_MAX, color="grey", ls=":",  lw=0.8)
    axes[0, 2].set_title("Altitude Profile")
    axes[0, 2].set_xlabel("Slot"); axes[0, 2].set_ylabel("m")
    axes[0, 2].legend(fontsize=8); axes[0, 2].grid(True, alpha=0.4)

    # Panel 4: task scheduling
    _draw_scheduling(axes[1, 0], omega, b, env, title="Task Scheduling")

    # Panel 5: alpha
    axes[1, 1].plot(range(Q), uav.alpha, color="#ff7f0e", lw=1.5)
    axes[1, 1].axhline(0.5, color="grey", ls="--", lw=0.8)
    axes[1, 1].set_title("Power Split α")
    axes[1, 1].set_xlabel("Slot"); axes[1, 1].set_ylabel("α")
    axes[1, 1].set_ylim(0, 1); axes[1, 1].grid(True, alpha=0.4)

    # Panel 6: speed
    axes[1, 2].plot(range(Q), np.linalg.norm(final_vel, axis=1),
                    color="#17becf", lw=1.5)
    axes[1, 2].set_title("UAV Speed")
    axes[1, 2].set_xlabel("Slot"); axes[1, 2].set_ylabel("m/s")
    axes[1, 2].grid(True, alpha=0.4)

    fig.tight_layout()
    _save(fig, "dashboard.png")


# =============================================================================
# SECTION 1 — Common environment
# =============================================================================
print("=" * 65)
print("  UAV-ISAC 3-D Trajectory Optimisation")
print("  (Liu et al., IEEE TWC 2024)  —  Baseline + Improved")
print("=" * 65)

env = Environment(num_nodes=12, area_size=1200, seed=42)
env.print_state()
print()


# =============================================================================
# SECTION 2 — BASELINE (original paper Algorithm 2)
# =============================================================================
print("-" * 65)
print("  BASELINE  -  Single-objective, no fairness constraint")
print("-" * 65)

uav_b = UAV(Q=200, T=100.0)
uav_b.initialize_trajectory(env)
init_pos_b = uav_b.position.copy()
init_vel_b = uav_b.velocity.copy()

hk_com0, hk_rad0, hc0, _, _ = _channel_gains(uav_b, env)
omega0, b0, _, _, _ = solve_scheduling(uav_b, env, hk_com0, hk_rad0, hc0)
print(f"\n  Initial Radar Rate : {compute_rate_baseline(uav_b, env, omega0):.4f} bps/Hz\n")

rates_b, omega_hist_b = three_layer_optimize(
    uav_b, env, max_outer=20, tol=1e-3, verbose=True)

omega_b = omega_hist_b[-1]
hk_com_f, hk_rad_f, hc_f, _, _ = _channel_gains(uav_b, env)
_, b_b, _, _, _ = solve_scheduling(uav_b, env, hk_com_f, hk_rad_f, hc_f)

energy_b = float(np.sum(uav_b.Pt) * uav_b.dt)
print(f"\n  Final Radar Rate : {rates_b[-1]:.4f} bps/Hz")
print(f"  Total Energy     : {energy_b:.2f} J")

print("\n  Generating baseline individual plots...")
generate_individual_plots(
    "output_plots_baseline", "Baseline",
    init_pos_b, uav_b.position,
    init_vel_b, uav_b.velocity,
    uav_b, env, rates_b, omega_b, b_b)


# =============================================================================
# SECTION 3 — IMPROVED (fairness + multi-objective)
# =============================================================================
print("\n" + "-" * 65)
print(f"  IMPROVED  -  Fairness (R_min={R_MIN}) + Pareto (lam={LAM})")
print("-" * 65)

uav_i = UAV(Q=200, T=100.0)
uav_i.initialize_trajectory(env)
init_pos_i = uav_i.position.copy()
init_vel_i = uav_i.velocity.copy()

rates_i = []
omega_i = b_i = ns_i = None
prev_comp = -np.inf

print()
for i in range(20):
    hk_com, hk_rad, hc, _, _ = _channel_gains_imp(uav_i, env)
    omega_i, b_i, Rrad, Rcom, Rc, ns_i = solve_scheduling_fair(
        uav_i, env, hk_com, hk_rad, hc, R_min=R_MIN)
    solve_power_multiobjective(uav_i, env, omega_i, b_i, hk_com, hk_rad, hc,
                               lam=LAM, max_iter=30)
    solve_trajectory(uav_i, env, omega_i, b_i, node_service=ns_i, R_min=R_MIN)

    rate_i   = compute_rate_improved(uav_i, env, omega_i)
    energy_i = compute_energy(uav_i)
    rates_i.append(rate_i)

    fair = int(np.sum(ns_i >= R_MIN))
    print(f"  Iter {i+1:2d}  |  Radar: {rate_i:8.4f} bps/Hz"
          f"  |  Energy: {energy_i:7.2f} J"
          f"  |  Fair nodes: {fair}/{env.num_nodes}")

    e_scale = uav_i.Q * uav_i.P_AVG * uav_i.dt
    composite = LAM * rate_i - (1 - LAM) * (energy_i / (e_scale + 1e-12))
    if abs(composite - prev_comp) < 1e-3 and i > 1:
        print(f"  Converged at iteration {i+1}")
        break
    prev_comp = composite

energy_i_final = compute_energy(uav_i)
print(f"\n  Final Radar Rate : {rates_i[-1]:.4f} bps/Hz")
print(f"  Total Energy     : {energy_i_final:.2f} J")

print("\n  Generating improved individual plots...")
generate_individual_plots(
    "output_plots_improved", "Improved",
    init_pos_i, uav_i.position,
    init_vel_i, uav_i.velocity,
    uav_i, env, rates_i, omega_i, b_i)


# =============================================================================
# SECTION 4 — Comparison plots  (4 figures, paper-style)
# =============================================================================
print("\n" + "-" * 65)
print("  COMPARISON PLOTS")
print("-" * 65)

Q = uav_b.Q
t = np.linspace(0, uav_b.T, Q)


# ── Fig 1: Convergence comparison  (like Fig. 7 / 12 of paper) ───────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, len(rates_b) + 1), rates_b,
        color=C_BLUE, lw=2.5, marker="o", ms=6,
        label="Baseline (single-obj, no fairness)")
ax.plot(range(1, len(rates_i) + 1), rates_i,
        color=C_GREEN, lw=2.5, marker="s", ms=6, ls="--",
        label=f"Improved (lam={LAM}, R_min={R_MIN})")
ax.set_xlabel("Number of Iterations", fontsize=13)
ax.set_ylabel("Sum Radar Estimation Rate (bps/Hz)", fontsize=13)
ax.set_title("Convergence Comparison: Baseline vs. Improved", fontsize=14)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
fig.tight_layout()
path = "output_plots_comparison/fig1_convergence_comparison.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"   Saved -> {path}")


# ── Fig 2: Power allocation comparison  (like Fig. 8 of paper) ───────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
fig.suptitle("Power Allocation: Baseline vs. Improved",
             fontsize=14, fontweight="bold")

# Top-left: Baseline stacked power bar
axes[0, 0].bar(range(Q), uav_b.Pcom, label="P_com", color=C_BLUE,  width=1.0)
axes[0, 0].bar(range(Q), uav_b.Prad, bottom=uav_b.Pcom,
               label="P_rad", color=C_RED, width=1.0)
axes[0, 0].set_ylabel("Power (W)", fontsize=11)
axes[0, 0].set_title("Baseline – Power Split (P_com / P_rad)", fontsize=12)
axes[0, 0].legend(fontsize=9); axes[0, 0].grid(axis="y", alpha=0.4)

# Top-right: Improved stacked power bar
axes[0, 1].bar(range(Q), uav_i.Pcom, label="P_com", color=C_BLUE,  width=1.0, alpha=0.85)
axes[0, 1].bar(range(Q), uav_i.Prad, bottom=uav_i.Pcom,
               label="P_rad", color=C_RED, width=1.0, alpha=0.85)
axes[0, 1].set_ylabel("Power (W)", fontsize=11)
axes[0, 1].set_title("Improved – Power Split (P_com / P_rad)", fontsize=12)
axes[0, 1].legend(fontsize=9); axes[0, 1].grid(axis="y", alpha=0.4)

# Bottom-left: Baseline alpha
axes[1, 0].plot(t, uav_b.alpha, color=C_BLUE, lw=1.5, label="Baseline alpha")
axes[1, 0].axhline(0.5, color="grey", ls="--", lw=0.8)
axes[1, 0].set_xlabel("Time (s)", fontsize=11); axes[1, 0].set_ylabel("alpha", fontsize=11)
axes[1, 0].set_ylim(0, 1)
axes[1, 0].set_title("Baseline – Power Split Factor alpha", fontsize=12)
axes[1, 0].legend(fontsize=9); axes[1, 0].grid(True, alpha=0.4)

# Bottom-right: both alpha overlaid
axes[1, 1].plot(t, uav_b.alpha, color=C_BLUE,  lw=1.5,       label="Baseline alpha")
axes[1, 1].plot(t, uav_i.alpha, color=C_GREEN, lw=1.5, ls="--", label="Improved alpha")
axes[1, 1].axhline(0.5, color="grey", ls="--", lw=0.8)
axes[1, 1].set_xlabel("Time (s)", fontsize=11); axes[1, 1].set_ylabel("alpha", fontsize=11)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].set_title("alpha Comparison (Baseline vs. Improved)", fontsize=12)
axes[1, 1].legend(fontsize=9); axes[1, 1].grid(True, alpha=0.4)

fig.tight_layout()
path = "output_plots_comparison/fig2_power_comparison.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"   Saved -> {path}")


# ── Fig 3: Task scheduling comparison  (like Fig. 4 of paper) ────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("UAV Task Scheduling: Baseline vs. Improved",
             fontsize=14, fontweight="bold")
_draw_scheduling(axes[0], omega_b, b_b, env,
                 title="Baseline – UAV Task Scheduling per Time Slot")
_draw_scheduling(axes[1], omega_i, b_i, env,
                 title="Improved – UAV Task Scheduling per Time Slot")
fig.tight_layout()
path = "output_plots_comparison/fig3_task_scheduling_comparison.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved -> {path}")


# ── Fig 4: Trajectory comparison  (like Fig. 3 of paper) ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("UAV Top-View Trajectory: Baseline vs. Improved",
             fontsize=14, fontweight="bold")
_draw_top_view(axes[0], init_pos_b, uav_b.position, env, omega_b,
               title="Baseline – Top-View Trajectory")
_draw_top_view(axes[1], init_pos_i, uav_i.position, env, omega_i,
               title="Improved – Top-View Trajectory")
fig.tight_layout()
path = "output_plots_comparison/fig4_trajectory_comparison.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved -> {path}")


# =============================================================================
# SECTION 5 — Summary
# =============================================================================
delta_rate   = rates_i[-1] - rates_b[-1]
delta_energy = energy_i_final - energy_b

print("\n" + "=" * 65)
print("  FINAL COMPARISON SUMMARY")
print("=" * 65)
print(f"  Radar Rate  |  Baseline: {rates_b[-1]:.4f}  "
      f"Improved: {rates_i[-1]:.4f}  delta={delta_rate:+.4f} bps/Hz")
print(f"  Energy      |  Baseline: {energy_b:.2f} J  "
      f"Improved: {energy_i_final:.2f} J  delta={delta_energy:+.2f} J")
print("=" * 65)
print()
print("Simulation complete!")
print("   output_plots_baseline/   <- baseline individual plots (8 files)")
print("   output_plots_improved/   <- improved individual plots (8 files)")
print("   output_plots_comparison/ <- 4 comparison figures")
print("=" * 65)
