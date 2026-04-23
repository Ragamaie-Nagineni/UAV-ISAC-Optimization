"""
run_baseline.py
===============
Runs the original paper algorithm (Algorithm 2, single-objective radar rate
maximisation, no fairness constraint) and saves results to results_baseline.npz
for comparison with the improved version.
"""

import numpy as np
import os

from environment import Environment
from uav import UAV
from optimization_baseline import (
    three_layer_optimize,
    compute_total_radar_rate,
    _channel_gains,
    solve_scheduling,
)

OUTPUT_DIR = "output_plots_baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  BASELINE: UAV-ISAC (Liu et al., IEEE TWC 2024)")
print("  Single-objective radar rate maximisation")
print("=" * 60)

env = Environment(num_nodes=12, area_size=1200, seed=42)
uav = UAV(Q=200, T=100.0)

uav.initialize_trajectory(env)

# Initial state
hk_com0, hk_rad0, hc0, _, _ = _channel_gains(uav, env)
omega0, b0, _, _, _ = solve_scheduling(uav, env, hk_com0, hk_rad0, hc0)
initial_rate = compute_total_radar_rate(uav, env, omega0)
print(f"\n📶 Initial Sum Radar Rate : {initial_rate:.4f} bps/Hz\n")

print("🚀 Running baseline optimisation...")
print("-" * 60)
rates, omega_hist = three_layer_optimize(uav, env, max_outer=20, tol=1e-3, verbose=True)
print("-" * 60)

final_rate = rates[-1]
print(f"\n📶 Initial Rate : {initial_rate:.4f} bps/Hz")
print(f"📶 Final Rate   : {final_rate:.4f} bps/Hz")
print(f"📈 Improvement  : {(final_rate - initial_rate) / (abs(initial_rate) + 1e-12) * 100:+.1f}%")

# Per-node service
omega_final = omega_hist[-1]
from utils import h_rad, sinr_rad, radar_rate, distance_3d
node_service = np.zeros(env.num_nodes)
for q in range(uav.Q):
    for k in range(env.num_nodes):
        if omega_final[q, k] > 0.5:
            d = distance_3d(uav.position[q], env.node_pos3d(k))
            sr = sinr_rad(uav.Prad[q], uav.Pcom[q], h_rad(d))
            node_service[k] += radar_rate(sr)

# Estimate energy from power
energy = float(np.sum(uav.Pt) * uav.dt)

print(f"\n⚡ Total Energy  : {energy:.2f} J")
print(f"\n📊 Per-node radar service (bps/Hz):")
for k in range(env.num_nodes):
    print(f"   Node {k+1:2d}: {node_service[k]:.4f}")

# Save results
np.savez("results_baseline.npz",
         rates=np.array(rates),
         energy=energy,
         node_service=node_service,
         final_position=uav.position,
         omega_final=omega_final,
         Pt=uav.Pt,
         alpha=uav.alpha)

print("\n✅ Baseline results saved to results_baseline.npz")

# Generate baseline plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Convergence plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(rates) + 1), rates, "b-o", linewidth=2, markersize=6,
        label="Baseline (single-objective)")
ax.set_xlabel("Outer Iteration", fontsize=12)
ax.set_ylabel("Sum Radar Estimation Rate (bps/Hz)", fontsize=12)
ax.set_title("Baseline Convergence", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/convergence_baseline.png", dpi=150)
plt.close(fig)

# Per-node service bar chart
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#2196F3"] * env.num_nodes
ax.bar(range(1, env.num_nodes + 1), node_service, color=colors, edgecolor="black", alpha=0.85)
ax.set_xlabel("Node Index", fontsize=12)
ax.set_ylabel("Total Radar Service (bps/Hz)", fontsize=12)
ax.set_title("Baseline: Per-Node Radar Service (No Fairness Constraint)", fontsize=13)
ax.set_xticks(range(1, env.num_nodes + 1))
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/per_node_service_baseline.png", dpi=150)
plt.close(fig)

print(f"📊 Plots saved to {OUTPUT_DIR}/")
print("=" * 60)
