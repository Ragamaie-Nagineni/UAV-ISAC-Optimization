"""
main.py
=======
Entry point for the UAV-ISAC optimisation simulation.

Implements the system from:
  Liu et al., "UAV Assisted Integrated Sensing and Communications for IoT:
  3D Trajectory Optimization and Resource Allocation," IEEE TWC 2024.

This version runs BOTH:
  ① Baseline  – original Algorithm 2 (single-objective, no fairness)
  ② Improved  – multi-objective Pareto + min-rate fairness constraint

Then generates all individual plots AND a side-by-side comparison dashboard.

Workflow
--------
1. Initialise environment (IoT nodes + data-collection centre).
2. Run BASELINE: Algorithm 2 (radar rate maximisation only).
3. Run IMPROVED: fairness-aware scheduling + Pareto power allocation.
4. Compute Pareto front by sweeping transmit-power budget.
5. Report results and generate all plots.
"""

import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    solve_power_multiobjective,   # FIX: use the actual improved power solver
    solve_trajectory,
    compute_total_radar_rate as compute_rate_improved,
    compute_per_node_rate,
    compute_energy,
)

from utils import h_rad, sinr_rad, radar_rate, distance_3d

# ── Visualization imports (original paper plots) ─────────────────────────────
from visualization import (
    plot_3d_trajectory,
    plot_top_view,
    plot_convergence,
    plot_scheduling,
    plot_altitude,
    plot_speed,
    plot_power,
    plot_dashboard,
)

# ─────────────────────────────────────────────────────────────────────────────
# Directories
# ─────────────────────────────────────────────────────────────────────────────
for d in ["output_plots", "output_plots_baseline",
          "output_plots_improved", "output_plots_comparison"]:
    os.makedirs(d, exist_ok=True)

R_MIN = 0.5   # fairness floor (bps/Hz per node)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Jain's fairness index
# ─────────────────────────────────────────────────────────────────────────────
def jain_index(x):
    x = np.array(x, dtype=float)
    x = x[x > 0]
    if len(x) == 0:
        return 0.0
    return (np.sum(x) ** 2) / (len(x) * np.sum(x ** 2))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Common environment
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  UAV-ISAC 3-D Trajectory Optimisation")
print("  (Liu et al., IEEE TWC 2024)  —  Baseline + Improved")
print("=" * 65)

env = Environment(num_nodes=12, area_size=1200, seed=42)
env.print_state()
print()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BASELINE (original paper Algorithm 2)
# ═════════════════════════════════════════════════════════════════════════════
print("─" * 65)
print("  ① BASELINE  —  Single-objective, no fairness constraint")
print("─" * 65)

uav_b = UAV(Q=200, T=100.0)
uav_b.initialize_trajectory(env)
initial_position_b = uav_b.position.copy()
initial_velocity_b = uav_b.velocity.copy()

hk_com0, hk_rad0, hc0, _, _ = _channel_gains(uav_b, env)
omega0, b0, _, _, _ = solve_scheduling(uav_b, env, hk_com0, hk_rad0, hc0)
initial_rate_b = compute_rate_baseline(uav_b, env, omega0)
print(f"\n  📶 Initial Radar Rate : {initial_rate_b:.4f} bps/Hz\n")

rates_b, omega_hist_b = three_layer_optimize(
    uav_b, env, max_outer=20, tol=1e-3, verbose=True)

omega_final_b = omega_hist_b[-1]
hk_com_f, hk_rad_f, hc_f, _, _ = _channel_gains(uav_b, env)
_, b_final_b, _, _, _ = solve_scheduling(uav_b, env, hk_com_f, hk_rad_f, hc_f)

# Per-node service
node_svc_b = np.zeros(env.num_nodes)
for q in range(uav_b.Q):
    for k in range(env.num_nodes):
        if omega_final_b[q, k] > 0.5:
            d = distance_3d(uav_b.position[q], env.node_pos3d(k))
            sr = sinr_rad(uav_b.Prad[q], uav_b.Pcom[q], h_rad(d))
            node_svc_b[k] += radar_rate(sr)

energy_b = float(np.sum(uav_b.Pt) * uav_b.dt)

print(f"\n  📶 Final Radar Rate : {rates_b[-1]:.4f} bps/Hz")
print(f"  ⚡ Total Energy     : {energy_b:.2f} J")
print(f"  📊 Per-node service (bps/Hz):")
for k in range(env.num_nodes):
    flag = "⚠️ " if node_svc_b[k] < R_MIN else "  "
    print(f"     Node {k+1:2d}: {node_svc_b[k]:.4f}  {flag}")
print(f"  🎯 Nodes ≥ R_min={R_MIN}: "
      f"{int(np.sum(node_svc_b >= R_MIN))}/{env.num_nodes}")

# ── Baseline original plots (output_plots/) ────────────────────────────────
print("\n  📊 Generating baseline paper-style plots...")
plot_3d_trajectory(initial_position_b, uav_b.position, env, omega=omega_final_b)
plot_top_view(initial_position_b, uav_b.position, env, omega=omega_final_b)
plot_convergence(rates_b)
plot_scheduling(omega_final_b, b_final_b, env)
plot_altitude(initial_position_b, uav_b.position)
plot_speed(initial_velocity_b, uav_b.velocity)
plot_power(uav_b, omega_final_b, b_final_b)
plot_dashboard(initial_position_b, uav_b.position, uav_b, env,
               rates_b, omega_final_b, b_final_b)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — IMPROVED (fairness + Pareto multi-objective)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
LAM = 1.0   # Pareto weight: 1.0 = pure radar maximisation (best rate + fairness)
            # Try 0.8 for a slight energy reduction at small rate cost

print(f"  ② IMPROVED  —  Fairness (R_min={R_MIN}) + Pareto (λ={LAM})")
print("─" * 65)

uav_i = UAV(Q=200, T=100.0)
uav_i.initialize_trajectory(env)
initial_position_i = uav_i.position.copy()
initial_velocity_i = uav_i.velocity.copy()

rates_i    = []
energies_i = []
prev_rate  = -np.inf

print()
for i in range(20):
    hk_com, hk_rad, hc, _, _ = _channel_gains_imp(uav_i, env)
    # FIX A: Use the actual improved fairness-aware scheduling
    omega_i, b_i, Rrad, Rcom, Rc, ns = solve_scheduling_fair(
        uav_i, env, hk_com, hk_rad, hc, R_min=R_MIN)
    # FIX B: Use the actual improved multi-objective power solver (not baseline's)
    solve_power_multiobjective(uav_i, env, omega_i, b_i, hk_com, hk_rad, hc,
                               lam=LAM, max_iter=30)
    # FIX C: Pass node_service so trajectory flies lower over deprived nodes
    solve_trajectory(uav_i, env, omega_i, b_i, node_service=ns, R_min=R_MIN)

    rate_i   = compute_rate_improved(uav_i, env, omega_i)
    energy_i = compute_energy(uav_i)
    rates_i.append(rate_i)
    energies_i.append(energy_i)

    fair = int(np.sum(ns >= R_MIN))
    print(f"  Iter {i+1:2d}  |  Radar: {rate_i:8.4f} bps/Hz"
          f"  |  Energy: {energy_i:7.2f} J"
          f"  |  Fair nodes: {fair}/{env.num_nodes}")

    if abs(rate_i - prev_rate) < 1e-3 and i > 1:
        print(f"  ✅ Converged at iteration {i+1}")
        break
    prev_rate = rate_i

node_svc_i = compute_per_node_rate(uav_i, env, omega_i)
energy_i_final = energies_i[-1]

print(f"\n  📶 Final Radar Rate : {rates_i[-1]:.4f} bps/Hz")
print(f"  ⚡ Total Energy     : {energy_i_final:.2f} J")
print(f"  📊 Per-node service (bps/Hz):")
for k in range(env.num_nodes):
    flag = "✅" if node_svc_i[k] >= R_MIN else "⚠️ "
    print(f"     Node {k+1:2d}: {node_svc_i[k]:.4f}  {flag}")
print(f"  🎯 Nodes ≥ R_min={R_MIN}: "
      f"{int(np.sum(node_svc_i >= R_MIN))}/{env.num_nodes}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Pareto front sweep
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("  ③ PARETO FRONT  —  Sweeping transmit-power budget")
print("─" * 65)

p_avgs      = np.array([0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0])
lam_vals    = np.linspace(0.0, 1.0, len(p_avgs))
pareto_r    = []
pareto_e    = []

for p_avg, lam in zip(p_avgs, lam_vals):
    env_p = Environment(num_nodes=12, area_size=1200, seed=42)
    uav_p = UAV(Q=200, T=100.0)
    uav_p.P_AVG = p_avg
    uav_p.Pt[:] = p_avg
    uav_p.initialize_trajectory(env_p)
    for _ in range(5):
        hk_com, hk_rad, hc, _, _ = _channel_gains_imp(uav_p, env_p)
        om_p, b_p, _, _, _, _ = solve_scheduling_fair(
            uav_p, env_p, hk_com, hk_rad, hc, R_min=R_MIN)
        solve_trajectory(uav_p, env_p, om_p, b_p)
    r_p = compute_rate_improved(uav_p, env_p, om_p)
    e_p = float(np.sum(uav_p.Pt) * uav_p.dt)
    pareto_r.append(r_p)
    pareto_e.append(e_p)
    print(f"  λ={lam:.2f}  P_avg={p_avg:.2f} W  →  R={r_p:.4f} bps/Hz  E={e_p:.2f} J")

pareto_r = np.array(pareto_r)
pareto_e = np.array(pareto_e)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Final summary
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  FINAL COMPARISON SUMMARY")
print("=" * 65)
delta_rate   = rates_i[-1] - rates_b[-1]
delta_energy = energy_i_final - energy_b
jain_b = jain_index(node_svc_b)
jain_i = jain_index(node_svc_i)
n_fair_b = int(np.sum(node_svc_b >= R_MIN))
n_fair_i = int(np.sum(node_svc_i >= R_MIN))

print(f"  Radar Rate   |  Baseline: {rates_b[-1]:.4f}   "
      f"Improved: {rates_i[-1]:.4f}   Δ={delta_rate:+.4f} bps/Hz")
print(f"  Energy       |  Baseline: {energy_b:.2f} J   "
      f"Improved: {energy_i_final:.2f} J   Δ={delta_energy:+.2f} J")
print(f"  Fair Nodes   |  Baseline: {n_fair_b}/{env.num_nodes}          "
      f"Improved: {n_fair_i}/{env.num_nodes}")
print(f"  Jain Index   |  Baseline: {jain_b:.4f}    "
      f"Improved: {jain_i:.4f}    Δ={jain_i - jain_b:+.4f}")
print("=" * 65)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Comparison plots
# ═════════════════════════════════════════════════════════════════════════════
print("\n  📊 Generating comparison plots...")

BLUE   = "#1565C0"
GREEN  = "#2E7D32"
RED    = "#C62828"

# ── Fig 1: Convergence ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, len(rates_b) + 1), rates_b, color=BLUE, lw=2.5,
        marker="o", ms=6, label="Baseline (single-obj, no fairness)")
ax.plot(range(1, len(rates_i) + 1), rates_i, color=GREEN, lw=2.5,
        marker="s", ms=6, ls="--", label=f"Improved (λ=0.8, R_min={R_MIN})")
ax.set_xlabel("Outer Iteration", fontsize=13)
ax.set_ylabel("Sum Radar Estimation Rate (bps/Hz)", fontsize=13)
ax.set_title("Convergence: Baseline vs. Improved", fontsize=14)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("output_plots_comparison/fig1_convergence_comparison.png", dpi=150)
plt.close(fig)

# ── Fig 2: Per-node service ──────────────────────────────────────────────────
K = env.num_nodes
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(1, K + 1)
w = 0.38
bars_b = ax.bar(x - w / 2, node_svc_b, w, color=BLUE, alpha=0.80,
                edgecolor="black", lw=0.8, label="Baseline")
bars_i = ax.bar(x + w / 2, node_svc_i, w, color=GREEN, alpha=0.80,
                edgecolor="black", lw=0.8, label="Improved (fairness)")
# Highlight baseline nodes below R_min in red
for k in range(K):
    if node_svc_b[k] < R_MIN:
        ax.bar(k + 1 - w / 2, node_svc_b[k], w, color=RED, alpha=0.85,
               edgecolor="black", lw=0.8)
ax.axhline(R_MIN, color=RED, ls="--", lw=1.8, label=f"R_min = {R_MIN} bps/Hz")
ax.set_xlabel("Node Index", fontsize=13)
ax.set_ylabel("Total Radar Service (bps/Hz)", fontsize=13)
ax.set_title("Per-Node Radar Service: Baseline vs. Improved", fontsize=14)
ax.set_xticks(x); ax.legend(fontsize=11); ax.grid(True, axis="y", alpha=0.3)
n_below_b = int(np.sum(node_svc_b < R_MIN))
n_below_i = int(np.sum(node_svc_i < R_MIN))
ax.text(0.02, 0.96,
        f"Nodes below R_min — Baseline: {n_below_b}   Improved: {n_below_i}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
fig.tight_layout()
fig.savefig("output_plots_comparison/fig2_per_node_service.png", dpi=150)
plt.close(fig)

# ── Fig 3: Fairness metrics ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
bars = ax.bar(["Baseline", "Improved"], [jain_b, jain_i],
              color=[BLUE, GREEN], edgecolor="black", alpha=0.85, width=0.4)
ax.set_ylim(0, 1.1)
ax.axhline(1.0, color="gray", ls=":", lw=1.5, label="Perfect fairness")
for bar, val in zip(bars, [jain_b, jain_i]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.4f}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Jain's Fairness Index", fontsize=12)
ax.set_title("Jain's Fairness Index\n(higher = fairer)", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)

ax = axes[1]
labels = ["Min", "Mean", "Max"]
stats_b = [node_svc_b.min(), node_svc_b.mean(), node_svc_b.max()]
stats_i = [node_svc_i.min(), node_svc_i.mean(), node_svc_i.max()]
x2 = np.arange(len(labels)); w2 = 0.35
ax.bar(x2 - w2 / 2, stats_b, w2, color=BLUE, alpha=0.85, edgecolor="black",
       label="Baseline")
ax.bar(x2 + w2 / 2, stats_i, w2, color=GREEN, alpha=0.85, edgecolor="black",
       label="Improved")
ax.axhline(R_MIN, color=RED, ls="--", lw=1.8, label=f"R_min={R_MIN}")
ax.set_xticks(x2); ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Radar Service (bps/Hz)", fontsize=12)
ax.set_title("Node Service Statistics\n(min / mean / max)", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)
fig.suptitle("Fairness Metrics: Baseline vs. Improved", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("output_plots_comparison/fig3_fairness_metrics.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)

# ── Fig 4: Pareto front ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(pareto_e, pareto_r, c=lam_vals, cmap="plasma",
                s=140, zorder=5, edgecolors="black", lw=0.8)
ax.plot(pareto_e, pareto_r, "k--", lw=1.2, alpha=0.5, zorder=4)
for lv, en, ra in zip(lam_vals, pareto_e, pareto_r):
    ax.annotate(f"λ={lv:.1f}", (en, ra),
                textcoords="offset points", xytext=(6, 4), fontsize=8.5)
ax.scatter([energy_b], [rates_b[-1]], marker="*", s=280, color=RED, zorder=6,
           label=f"Baseline (no fairness)\nR={rates_b[-1]:.3f}, E={energy_b:.1f} J")
plt.colorbar(sc, ax=ax, label="Pareto Weight λ")
ax.set_xlabel("Total Energy Consumption (J)", fontsize=13)
ax.set_ylabel("Sum Radar Estimation Rate (bps/Hz)", fontsize=13)
ax.set_title("Pareto Front: Radar Rate vs. Energy\n(with Fairness Constraint Active)",
             fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("output_plots_comparison/fig4_pareto_front.png", dpi=150)
plt.close(fig)

# ── Fig 5: Power profile comparison ─────────────────────────────────────────
Q = uav_b.Q
t = np.linspace(0, uav_b.T, Q)
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(t, uav_b.Pt, color=BLUE, lw=1.5, alpha=0.85, label="Baseline Pt")
axes[0].plot(t, uav_i.Pt, color=GREEN, lw=1.5, alpha=0.85, ls="--",
             label="Improved Pt")
axes[0].set_ylabel("Total Transmit Power (W)", fontsize=12)
axes[0].set_title("Power Allocation: Baseline vs. Improved", fontsize=13)
axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)
axes[1].plot(t, uav_b.alpha, color=BLUE, lw=1.5, alpha=0.85,
             label="Baseline α (comm fraction)")
axes[1].plot(t, uav_i.alpha, color=GREEN, lw=1.5, alpha=0.85, ls="--",
             label="Improved α (comm fraction)")
axes[1].set_xlabel("Time (s)", fontsize=12)
axes[1].set_ylabel("Power Split α", fontsize=12)
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("output_plots_comparison/fig5_power_comparison.png", dpi=150)
plt.close(fig)

# ── Fig 6: Full dashboard ────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

ax = fig.add_subplot(gs[0, 0])
ax.plot(range(1, len(rates_b) + 1), rates_b, color=BLUE, lw=2,
        marker="o", ms=4, label="Baseline")
ax.plot(range(1, len(rates_i) + 1), rates_i, color=GREEN, lw=2,
        marker="s", ms=4, ls="--", label="Improved")
ax.set_xlabel("Iteration", fontsize=10); ax.set_ylabel("Radar Rate (bps/Hz)", fontsize=10)
ax.set_title("Convergence", fontsize=11); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
xp = np.arange(1, K + 1)
ax.bar(xp - 0.2, node_svc_b, 0.38, color=BLUE, alpha=0.80,
       edgecolor="black", lw=0.6, label="Baseline")
ax.bar(xp + 0.2, node_svc_i, 0.38, color=GREEN, alpha=0.80,
       edgecolor="black", lw=0.6, label="Improved")
ax.axhline(R_MIN, color=RED, ls="--", lw=1.5, label=f"R_min={R_MIN}")
ax.set_xlabel("Node", fontsize=10); ax.set_ylabel("Service (bps/Hz)", fontsize=10)
ax.set_title("Per-Node Service", fontsize=11)
ax.set_xticks(xp); ax.tick_params(axis="x", labelsize=7)
ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

ax = fig.add_subplot(gs[0, 2])
brs = ax.bar(["Baseline", "Improved"], [jain_b, jain_i],
             color=[BLUE, GREEN], edgecolor="black", alpha=0.85, width=0.4)
for bar, val in zip(brs, [jain_b, jain_i]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.1); ax.set_ylabel("Jain's Fairness Index", fontsize=10)
ax.set_title("Fairness Index", fontsize=11); ax.grid(True, axis="y", alpha=0.3)

ax = fig.add_subplot(gs[1, 0])
sc2 = ax.scatter(pareto_e, pareto_r, c=lam_vals, cmap="plasma",
                 s=80, zorder=5, edgecolors="black", lw=0.6)
ax.plot(pareto_e, pareto_r, "k--", lw=1, alpha=0.5, zorder=4)
ax.scatter([energy_b], [rates_b[-1]], marker="*", s=200,
           color=RED, zorder=6, label="Baseline")
plt.colorbar(sc2, ax=ax, label="λ")
ax.set_xlabel("Energy (J)", fontsize=10); ax.set_ylabel("Radar Rate (bps/Hz)", fontsize=10)
ax.set_title("Pareto Front", fontsize=11); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
brs2 = ax.bar(["Baseline", "Improved"], [energy_b, energy_i_final],
              color=[BLUE, GREEN], edgecolor="black", alpha=0.85, width=0.4)
for bar, val in zip(brs2, [energy_b, energy_i_final]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.2f} J", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Total Energy (J)", fontsize=10)
ax.set_title("Energy Consumption", fontsize=11); ax.grid(True, axis="y", alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
summary_text = (
    f"{'SUMMARY':^30}\n"
    f"{'─'*30}\n\n"
    f"  Radar Rate\n"
    f"    Baseline : {rates_b[-1]:.4f} bps/Hz\n"
    f"    Improved : {rates_i[-1]:.4f} bps/Hz\n"
    f"    Δ        : {delta_rate:+.4f} bps/Hz\n\n"
    f"  Energy\n"
    f"    Baseline : {energy_b:.2f} J\n"
    f"    Improved : {energy_i_final:.2f} J\n"
    f"    Δ        : {delta_energy:+.2f} J\n\n"
    f"  Fairness (R_min={R_MIN})\n"
    f"    Baseline : {n_fair_b}/{K} nodes\n"
    f"    Improved : {n_fair_i}/{K} nodes\n\n"
    f"  Jain Index\n"
    f"    Baseline : {jain_b:.4f}\n"
    f"    Improved : {jain_i:.4f}\n"
    f"    Δ        : {jain_i - jain_b:+.4f}"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=10, va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

fig.suptitle(
    "UAV-ISAC: Baseline vs. Improved — Full Comparison Dashboard",
    fontsize=14, fontweight="bold", y=1.01)
fig.savefig("output_plots_comparison/fig6_dashboard.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)

print("   💾  Saved → output_plots_comparison/fig1_convergence_comparison.png")
print("   💾  Saved → output_plots_comparison/fig2_per_node_service.png")
print("   💾  Saved → output_plots_comparison/fig3_fairness_metrics.png")
print("   💾  Saved → output_plots_comparison/fig4_pareto_front.png")
print("   💾  Saved → output_plots_comparison/fig5_power_comparison.png")
print("   💾  Saved → output_plots_comparison/fig6_dashboard.png")

print()
print("✅ Simulation complete!")
print("   output_plots/            ← baseline paper-style plots")
print("   output_plots_comparison/ ← baseline vs improved comparison plots")
print("=" * 65)
