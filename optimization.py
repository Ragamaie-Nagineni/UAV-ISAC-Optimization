import numpy as np;
from utils import distance, h_rad, rate;
def schedule(uav, env):
    schedule = []
    
    for q in range(uav.Q):
        uav_pos = uav.position[q]
        
        distances = [
            np.linalg.norm(uav_pos[:2] - node)
            for node in env.nodes
        ]
        
        nearest = np.argmin(distances)
        
        if distances[nearest] < 200:
            schedule.append(("ISAC", nearest))
        else:
            schedule.append(("UPLOAD", -1))
    
    return schedule

def allocate_power(task):
    if task == "ISAC":
        return 0.4
    return 1.0
from utils import distance, h_rad, rate
import numpy as np

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
            
            sinr = (Pr * h_rad(d)) / (Pc * h_rad(d) + 1e-9)
            total_rate += rate(sinr)
    
    return total_rate


def update_trajectory(uav, env):
    for q in range(uav.Q):
        random_node = env.nodes[np.random.randint(env.num_nodes)]
        direction = np.append(random_node, 100) - uav.position[q]
        uav.position[q] += 0.05 * direction