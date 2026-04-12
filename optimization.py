import numpy as np
from utils import distance, h_rad, rate


# ------------------ 1. SMART SCHEDULING ------------------
def schedule(uav, env):
    schedule = []
    
    for q in range(uav.Q):
        uav_pos = uav.position[q]
        
        # Distance to all nodes
        distances = np.linalg.norm(env.nodes - uav_pos[:2], axis=1)
        
        # 🔥 Choose among top 3 nearest nodes (not always same)
        sorted_nodes = np.argsort(distances)
        nearest = np.random.choice(sorted_nodes[:3])
        
        # Dynamic threshold based on height
        threshold = 200 + 0.2 * uav_pos[2]
        
        if distances[nearest] < threshold:
            schedule.append(("ISAC", nearest))
        else:
            schedule.append(("UPLOAD", -1))
    
    return schedule


# ------------------ 2. DYNAMIC POWER ALLOCATION ------------------
def allocate_power(task):
    if task == "ISAC":
        # More power to radar (dynamic)
        return np.random.uniform(0.3, 0.7)
    else:
        # Full communication power
        return np.random.uniform(0.8, 1.0)


# ------------------ 3. RADAR RATE COMPUTATION ------------------
def compute_radar_rate(uav, env, schedule_list):
    total_rate = 0
    
    for q in range(uav.Q):
        task, k = schedule_list[q]
        
        if task == "ISAC":
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
    step_size = 5  # smoother movement
    
    for q in range(uav.Q - 1):
        task, k = schedule_list[q]
        
        # Target selection
        if task == "ISAC":
            target = env.nodes[k]
        else:
            target = env.data_center
        
        target_3d = np.append(target, 100)
        
        direction = target_3d - uav.position[q]
        
        # Normalize direction
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        # 🔥 Smooth update (NOT aggressive jump)
        uav.position[q+1] = uav.position[q] + step_size * direction


# ------------------ 5. EXTRA: METRICS FOR CONSOLE ------------------
def compute_metrics(uav, env, schedule_list):
    isac_tasks = sum(1 for task, _ in schedule_list if task == "ISAC")
    upload_tasks = uav.Q - isac_tasks
    
    distances = []
    for q in range(uav.Q):
        task, k = schedule_list[q]
        if task == "ISAC":
            d = np.linalg.norm(uav.position[q][:2] - env.nodes[k])
            distances.append(d)
    
    avg_distance = np.mean(distances) if distances else 0
    avg_alpha = np.mean(uav.alpha)
    
    return isac_tasks, upload_tasks, avg_distance, avg_alpha