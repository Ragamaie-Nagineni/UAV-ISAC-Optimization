import numpy as np
from utils import distance, h_rad, rate


# ------------------ 1. SMART SCHEDULING ------------------
def schedule(uav, env):
    schedule = []

    for q in range(uav.Q):
        uav_pos = uav.position[q]

        # Distance to all nodes
        distances = np.linalg.norm(env.nodes - uav_pos[:2], axis=1)

        # Choose among nearest nodes
        sorted_nodes = np.argsort(distances)

        if np.random.rand() < 0.8:
            nearest = sorted_nodes[0]
        else:
            nearest = np.random.choice(sorted_nodes[:3])

        # ✅ Balanced condition (IMPORTANT FIX)
        if distances[nearest] < 200:
            if np.random.rand() > 0.2:
                schedule.append(("ISAC", nearest))
            else:
                schedule.append(("UPLOAD", -1))
        else:
            if np.random.rand() > 0.7:
                schedule.append(("ISAC", nearest))
            else:
                schedule.append(("UPLOAD", -1))

    return schedule


# ------------------ 2. DYNAMIC POWER ALLOCATION ------------------
def allocate_power(task):
    if task == "ISAC":
        return np.random.uniform(0.3, 0.6)
    else:
        return np.random.uniform(0.7, 1.0)


# ------------------ 3. RADAR RATE COMPUTATION ------------------
def compute_radar_rate(uav, env, schedule_list):
    total_rate = 0

    for q in range(len(schedule_list)):
        task, k = schedule_list[q]

        # ✅ ONLY compute when ISAC
        if task == "ISAC" and k != -1:
            uav_pos = uav.position[q]
            node_pos = np.append(env.nodes[k], 0)

            d = distance(uav_pos, node_pos)

            alpha = uav.alpha[q]
            Pc = alpha
            Pr = 1 - alpha

            # Radar SINR
            sinr = (Pr * h_rad(d)) / (Pc * h_rad(d) + 1e-9)

            total_rate += rate(sinr)

    return total_rate


# ------------------ 4. IMPROVED TRAJECTORY UPDATE ------------------
def update_trajectory(uav, env, schedule_list):
    for q in range(uav.Q - 1):
        task, k = schedule_list[q]

        # Target selection
        if task == "ISAC" and k != -1:
            if np.random.rand() < 0.6:
                target = env.nodes[k]
            else:
                # 🔥 exploration
                random_node = np.random.randint(0, len(env.nodes))
                target = env.nodes[random_node]
        else:
            target = env.data_center

        target_3d = np.append(target, 100)

        # Direction
        direction = target_3d - uav.position[q]
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        # Controlled randomness
        noise = np.random.randn(3) * 0.3
        direction = direction + noise
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        # Smooth motion
        inertia = 0.7

        uav.position[q+1] = (
            inertia * uav.position[q] +
            (1 - inertia) * (uav.position[q] + 10 * direction)
        )


# ------------------ 5. EXTRA: METRICS ------------------
def compute_metrics(uav, env, schedule_list):
    isac_tasks = sum(1 for task, _ in schedule_list if task == "ISAC")
    upload_tasks = len(schedule_list) - isac_tasks

    distances = []

    for q in range(len(schedule_list)):
        task, k = schedule_list[q]

        if task == "ISAC" and k != -1:
            d = np.linalg.norm(uav.position[q][:2] - env.nodes[k])
            distances.append(d)

    avg_distance = np.mean(distances) if distances else 0
    avg_alpha = np.mean(uav.alpha)

    return isac_tasks, upload_tasks, avg_distance, avg_alpha