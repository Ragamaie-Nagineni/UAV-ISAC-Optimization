from environment import Environment
from uav import UAV
from optimization import (
    schedule,
    allocate_power,
    compute_radar_rate,
    update_trajectory,
    compute_metrics
)
from visualization import (
    plot_all,
    plot_trajectory,
    plot_before_after,
    plot_rate,
    plot_schedule,
    plot_power,
    plot_speed,
    plot_3d_trajectory
)

import numpy as np


# ------------------ INITIALIZATION ------------------
env = Environment()
uav = UAV()

uav.initialize_trajectory(env.data_center)

# Save initial state
initial_trajectory = uav.position.copy()
rates = []
trajectory_history = []

print("\n🚀 Starting UAV ISAC Optimization...\n")
env.print_state()
uav.print_state()


# ------------------ ITERATIVE OPTIMIZATION ------------------
for i in range(20):

    # 🔥 Dynamic environment (nodes move)
    env.update_nodes()

    # Scheduling
    schedule_list = schedule(uav, env)

    # Power allocation
    for q in range(uav.Q):
        task, _ = schedule_list[q]
        uav.alpha[q] = allocate_power(task)

    # Compute radar rate
    rate = compute_radar_rate(uav, env, schedule_list)
    rates.append(rate)

    # 🔥 Compute additional metrics
    isac, upload, avg_dist, avg_alpha = compute_metrics(uav, env, schedule_list)

    # 🔥 Rich console output
    print(f"""
🔹 Iteration {i}
   Radar Rate        : {rate:.2f}
   ISAC Tasks        : {isac}
   Upload Tasks      : {upload}
   Avg UAV-Node Dist : {avg_dist:.2f}
   Avg Power (alpha) : {avg_alpha:.2f}
""")

    # Update UAV trajectory
    update_trajectory(uav, env, schedule_list)

    # Dynamic UAV behavior
    uav.adjust_height()
    uav.update_velocity()

    # Store history
    trajectory_history.append(uav.position.copy())


# ------------------ FINAL VISUALIZATION ------------------

print("\n📊 Generating Visualizations...\n")

# 🔥 ALL-IN-ONE DASHBOARD (BEST)
plot_all(
    trajectory_history,
    env,
    initial_trajectory,
    uav.position,
    rates,
    schedule_list,
    uav
)

# Optional individual plots (you can comment if not needed)
plot_trajectory(trajectory_history, env)
plot_before_after(initial_trajectory, uav.position, env)
plot_rate(rates)
plot_schedule(schedule_list)
plot_power(uav)
plot_speed(uav)
plot_3d_trajectory(uav, env)

print("\n✅ Simulation Complete!\n")