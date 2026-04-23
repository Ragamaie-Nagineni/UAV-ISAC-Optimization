"""
run_improved.py
===============
Runs the improved algorithm with:
  1. Multi-objective (Pareto) power allocation  (lambda sweep)
  2. Min-rate fairness constraint               (R_min = 0.5 bps/Hz)

Saves results to results_improved.npz for comparison.
"""

import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import Environment
from uav import UAV
from optimization_improved import (
    three_layer_optimize_improved,
    compute_pareto_front,
    compute_per_node_rate,
    compute_energy,
)

OUTPUT_DIR = "output_plots_improved"
os.makedirs(OUTPUT_DIR, exist_ok=True)

R_MIN = 0.5   # fairness floor [bps/Hz per node]

print("=" * 60)
print("  IMPROVED: UAV-ISAC with Multi-Objective + Fairness")
print(f"  R_min = {R_MIN} bps/Hz, lambda = 0.8 (main run)")
print("=" * 60)

env = Environment(num_nodes=12, area_size=1200, seed=42)
uav = UAV(Q=200, T=100.0)
uav.initialize_trajectory(env)

print("\n🚀 Running improved optimisation (λ=0.8, fairness ON)...")
print("-" * 60)

rates, energies, omega_hist, node_service = three_layer_optimize_improved(
    uav, env,
    max_outer=20,
    tol=1e-3,
    lam=0.8,
    R_min=R_MIN,
    verbose=True,
)
print("-" * 60)

final_rate   = rates[-1]
final_energy = energies[-1]

print(f"\n📶 Final Radar Rate : {final_rate:.4f} bps/Hz")
print(f"⚡ Total Energy      : {final_energy:.2f} J")
print(f"\n📊 Per-node radar service (bps/Hz):")
for k in range(env.num_nodes):
    flag = "✅" if node_service[k] >= R_MIN else "⚠️ "
    print(f"   Node {k+1:2d}: {node_service[k]:.4f}  {flag}")

fair_count = int(np.sum(node_service >= R_MIN))
print(f"\n🎯 Nodes meeting R_min={R_MIN}: {fair_count}/{env.num_nodes}")

# ── Pareto front sweep ──────────────────────────────────────────────────────
print("\n🔄 Computing Pareto front (lambda sweep 0→1)...")
lambdas = np.linspace(0.0, 1.0, 11)

pareto_rates, pareto_energies, lam_vals = compute_pareto_front(
    env_cls=Environment,
    uav_cls=UAV,
    env_kwargs=dict(num_nodes=12, area_size=1200, seed=42),
    uav_kwargs=dict(Q=200, T=100.0),
    lambdas=lambdas,
    R_min=R_MIN,
    max_outer=15,
    verbose=True,
)

# Save all results
np.savez("results_improved.npz",
         rates=np.array(rates),
         energies=np.array(energies),
         final_energy=final_energy,
         node_service=node_service,
         final_position=uav.position,
         omega_final=omega_hist[-1],
         Pt=uav.Pt,
         alpha=uav.alpha,
         pareto_rates=np.array(pareto_rates),
         pareto_energies=np.array(pareto_energies),
         lam_vals=np.array(lam_vals),
         R_min=R_MIN)

print("\n✅ Improved results saved to results_improved.npz")

# ── Plot 1: Convergence ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(rates) + 1), rates, "g-o", linewidth=2, markersize=6,
        label="Improved (λ=0.8, fairness)")
ax.set_xlabel("Outer Iteration", fontsize=12)
ax.set_ylabel("Sum Radar Estimation Rate (bps/Hz)", fontsize=12)
ax.set_title("Improved Algorithm Convergence", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/convergence_improved.png", dpi=150)
plt.close(fig)

# ── Plot 2: Per-node service ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#4CAF50" if s >= R_MIN else "#F44336" for s in node_service]
bars = ax.bar(range(1, env.num_nodes + 1), node_service, color=colors,
              edgecolor="black", alpha=0.85)
ax.axhline(R_MIN, color="red", linestyle="--", linewidth=1.8,
           label=f"R_min = {R_MIN} bps/Hz")
ax.set_xlabel("Node Index", fontsize=12)
ax.set_ylabel("Total Radar Service (bps/Hz)", fontsize=12)
ax.set_title("Improved: Per-Node Radar Service (With Fairness Constraint)", fontsize=13)
ax.set_xticks(range(1, env.num_nodes + 1))
ax.legend(fontsize=11)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/per_node_service_improved.png", dpi=150)
plt.close(fig)

# ── Plot 3: Pareto front ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(pareto_energies, pareto_rates, c=lam_vals,
                cmap="plasma", s=120, zorder=5, edgecolors="black", linewidths=0.8)
ax.plot(pareto_energies, pareto_rates, "k--", linewidth=1.2, alpha=0.5)

for lv, re, ra in zip(lam_vals, pareto_energies, pareto_rates):
    ax.annotate(f"λ={lv:.1f}", (re, ra),
                textcoords="offset points", xytext=(5, 5), fontsize=8)

cb = plt.colorbar(sc, ax=ax)
cb.set_label("Pareto Weight λ", fontsize=11)
ax.set_xlabel("Total Energy Consumption (J)", fontsize=12)
ax.set_ylabel("Sum Radar Estimation Rate (bps/Hz)", fontsize=12)
ax.set_title("Pareto Front: Radar Rate vs. Energy Consumption", fontsize=13)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/pareto_front.png", dpi=150)
plt.close(fig)

# ── Plot 4: Energy over iterations ──────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))
color1 = "#1565C0"
ax1.plot(range(1, len(rates) + 1), rates, color=color1, linewidth=2,
         marker="o", markersize=5, label="Radar Rate (bps/Hz)")
ax1.set_xlabel("Outer Iteration", fontsize=12)
ax1.set_ylabel("Radar Rate (bps/Hz)", color=color1, fontsize=12)
ax1.tick_params(axis="y", labelcolor=color1)

ax2 = ax1.twinx()
color2 = "#E65100"
ax2.plot(range(1, len(energies) + 1), energies, color=color2, linewidth=2,
         marker="s", markersize=5, linestyle="--", label="Energy (J)")
ax2.set_ylabel("Total Energy (J)", color=color2, fontsize=12)
ax2.tick_params(axis="y", labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center right")
ax1.set_title("Radar Rate and Energy over Iterations (Improved)", fontsize=13)
ax1.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/rate_energy_iterations.png", dpi=150)
plt.close(fig)

print(f"📊 All plots saved to {OUTPUT_DIR}/")
print("=" * 60)
