import numpy as np


class Environment:
    """
    Represents the IoT deployment area.

    Nodes and the data-collection centre are FIXED (as assumed in the paper).
    The area is 1200 m x 1200 m with K=12 IoT nodes randomly distributed
    and the data-collection centre at the centre.
    """

    def __init__(self, num_nodes=12, area_size=1200, seed=42):
        self.num_nodes = num_nodes
        self.area_size = area_size
        rng = np.random.default_rng(seed)

        # (K, 2) array of node (x,y) positions; altitude = 0
        self.nodes = rng.uniform(0, area_size, size=(num_nodes, 2))
        self.node_heights = np.zeros(num_nodes)          # ground level

        # Data-collection centre
        cx, cy = area_size / 2, area_size / 2
        self.data_center = np.array([cx, cy])
        self.data_center_height = 0.0                    # ground level

    # ------------------------------------------------------------------
    def node_pos3d(self, k):
        """3-D position of node k."""
        return np.array([self.nodes[k, 0], self.nodes[k, 1], self.node_heights[k]])

    def center_pos3d(self):
        """3-D position of the data-collection centre."""
        return np.array([self.data_center[0], self.data_center[1],
                         self.data_center_height])

    # ------------------------------------------------------------------
    def print_state(self):
        print("📡 Environment State:")
        print(f"   Number of Nodes : {self.num_nodes}")
        print(f"   Area Size       : {self.area_size} m × {self.area_size} m")
        print(f"   Data Centre     : {self.data_center}")
        print(f"   Node positions  :")
        for k, pos in enumerate(self.nodes):
            print(f"      Node {k+1:2d}: ({pos[0]:.1f}, {pos[1]:.1f})")
