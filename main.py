"""
main.py
=======
Entry point for the UAV-ISAC optimisation simulation.

Implements the system from:
  Liu et al., "UAV Assisted Integrated Sensing and Communications for IoT:
  3D Trajectory Optimization and Resource Allocation," IEEE TWC 2024.

Workflow
--------
1. Initialise environment (IoT nodes + data-collection centre).
2. Initialise UAV with circular baseline trajectory & equal power split.
3. Run Algorithm 2 (three-layer iterative optimisation).
4. Report results and generate all plots.
"""

import numpy as np

from environment import Environment
from uav import UAV
from optimization_baseline import three_layer_optimize, compute_total_radar_rate, _channel_gains, solve_scheduling
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
# 1. Setup
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  UAV-ISAC 3-D Trajectory Optimisation")
print("  (Liu et al., IEEE TWC 2024)")
print("=" * 60)

env = Environment(num_nodes=12, area_size=1200, seed=42)
uav = UAV(Q=200, T=100.0)           # Q=200 slots, T=100 s  (dt = 0.5 s)

env.print_state()
print()

uav.initialize_trajectory(env)
uav.print_state()
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Save initial state
# ─────────────────────────────────────────────────────────────────────────────

initial_position = uav.position.copy()
initial_velocity = uav.velocity.copy()

# Compute initial radar rate (with default power & scheduling)
hk_com0, hk_rad0, hc0, _, _ = _channel_gains(uav, env)
omega0, b0, _, _, _ = solve_scheduling(uav, env, hk_com0, hk_rad0, hc0)
initial_rate = compute_total_radar_rate(uav, env, omega0)
print(f"📶 Initial Sum Radar Estimation Rate : {initial_rate:.4f} bps/Hz")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Run Algorithm 2
# ─────────────────────────────────────────────────────────────────────────────

print("🚀 Running three-layer iterative optimisation (Algorithm 2)...")
print("-" * 60)

rates, omega_hist = three_layer_optimize(
    uav, env,
    max_outer=20,
    tol=1e-3,
    verbose=True
)

print("-" * 60)
final_rate = rates[-1]
print(f"\n📶 Initial Sum Radar Estimation Rate : {initial_rate:.4f} bps/Hz")
print(f"📶 Final   Sum Radar Estimation Rate : {final_rate:.4f} bps/Hz")
gain = (final_rate - initial_rate) / (abs(initial_rate) + 1e-12) * 100
print(f"📈 Improvement                       : {gain:+.1f}%")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Final scheduling for plots
# ─────────────────────────────────────────────────────────────────────────────

omega_final = omega_hist[-1]
hk_com_f, hk_rad_f, hc_f, _, _ = _channel_gains(uav, env)
_, b_final, _, _, _ = solve_scheduling(uav, env, hk_com_f, hk_rad_f, hc_f)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

print("📊 Generating plots...")

plot_3d_trajectory(initial_position, uav.position, env, omega=omega_final)
plot_top_view(initial_position, uav.position, env, omega=omega_final)
plot_convergence(rates)
plot_scheduling(omega_final, b_final, env)
plot_altitude(initial_position, uav.position)
plot_speed(initial_velocity, uav.velocity)
plot_power(uav, omega_final, b_final)
plot_dashboard(initial_position, uav.position, uav, env, rates, omega_final, b_final)

print()
print("✅ Simulation complete!  All plots saved to → output_plots/")
print("=" * 60)
