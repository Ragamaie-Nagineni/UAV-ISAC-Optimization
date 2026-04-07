import numpy as np

class UAV:
    def __init__(self, Q=100, T=100):
        self.Q = Q
        self.T = T
        self.dt = T / Q
        
        self.position = np.zeros((Q, 3))
        self.alpha = np.ones(Q) * 0.5 #create array and initialize all values to 0.5
        self.power = np.ones(Q)#all values to 1