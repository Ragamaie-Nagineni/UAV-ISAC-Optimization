import numpy as np;
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