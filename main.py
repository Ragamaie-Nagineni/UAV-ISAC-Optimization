from environment import Environment
from uav import UAV
from optimization import schedule, allocate_power, compute_radar_rate, update_trajectory
from visualization import plot_trajectory

# Initialize
env = Environment()
uav = UAV()

uav.initialize_trajectory(env.data_center)

# Iterative optimization
trajectory_history = []
for i in range(20):
    schedule_list = schedule(uav, env)
    
    # Power allocation
    for q in range(uav.Q):
        task, _ = schedule_list[q]
        uav.alpha[q] = allocate_power(task)
    
    rate = compute_radar_rate(uav, env, schedule_list)
    
    print(f"Iteration {i}, Radar Rate: {rate}")
    update_trajectory(uav, env, schedule_list)
    trajectory_history.append(uav.position.copy())

# Visualization
plot_trajectory(trajectory_history, env)