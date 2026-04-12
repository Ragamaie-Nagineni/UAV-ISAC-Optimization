import numpy as np

class Environment:
    def __init__(self, num_nodes=10, area_size=1000):
        self.num_nodes = num_nodes
        self.area_size = area_size
        
        # Random node positions (x, y)
        self.nodes = np.random.rand(num_nodes, 2) * area_size
        
        # Data collection center (middle of area)
        self.data_center = np.array([area_size / 2, area_size / 2])
    
    # ------------------  Dynamic Node Movement ------------------
    def update_nodes(self, step_size=5):
        """
        Slightly move nodes to simulate dynamic IoT environment
        """
        movement = np.random.randn(self.num_nodes, 2) * step_size
        self.nodes += movement
        
        # Keep nodes inside area bounds
        self.nodes = np.clip(self.nodes, 0, self.area_size)
    
    # ------------------  Distance Matrix (useful for analysis) ------------------
    def get_distance_matrix(self, uav_pos):
        """
        Compute distance from UAV to all nodes
        """
        return np.linalg.norm(self.nodes - uav_pos[:2], axis=1)
    
    # ------------------ Debug Info ------------------
    def print_state(self):
        """
        Print environment summary (for console richness)
        """
        print("📡 Environment State:")
        print(f"   Number of Nodes : {self.num_nodes}")
        print(f"   Area Size       : {self.area_size}")
        print(f"   Data Center     : {self.data_center}")
        print(f"   Sample Node     : {self.nodes[0]}")