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
  
