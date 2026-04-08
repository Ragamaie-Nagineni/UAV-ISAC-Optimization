import numpy as np

class UAV:
    def __init__(self, Q=100, T=100):
        self.Q = Q
        self.T = T
        self.dt = T / Q
        
        self.position = np.zeros((Q, 3))
        self.alpha = np.ones(Q) * 0.5 #create array and initialize all values to 0.5
        self.power = np.ones(Q)#all values to 1
        
    def initialize_trajectory(self, center, radius=300, height=100):
        for q in range(self.Q):
           angle = 2 * np.pi * q / self.Q
           self.position[q] = [
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            height
        ]
  

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