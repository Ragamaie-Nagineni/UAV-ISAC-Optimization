import numpy as np

class Environment:
    def __init__(self, num_nodes=10, area_size=1000):
        self.num_nodes = num_nodes
        self.area_size = area_size
        
        self.nodes = np.random.rand(num_nodes, 2) * area_size
        self.data_center = np.array([area_size/2, area_size/2])