import numpy as np

class UAV:
    def __init__(self, Q=100, T=100):
        self.Q = Q                  # number of time slots
        self.T = T                  # total time
        self.dt = T / Q             # time per slot
        
        # UAV state
        self.position = np.zeros((Q, 3))   # (x, y, z)
        self.alpha = np.ones(Q) * 0.5      # power split
        self.power = np.ones(Q)            # total power
        
        # 🔥 NEW: velocity tracking (for realism)
        self.velocity = np.zeros((Q, 3))
    
    # ------------------ 1. INITIAL TRAJECTORY ------------------
    def initialize_trajectory(self, center, radius=300, height=100):
        """
        Initialize circular trajectory (baseline from paper)
        """
        for q in range(self.Q):
            angle = 2 * np.pi * q / self.Q
            self.position[q] = [
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                height
            ]
    
    # ------------------ 2. UPDATE VELOCITY ------------------
    def update_velocity(self):
        """
        Compute velocity between time steps
        """
        for q in range(1, self.Q):
            self.velocity[q] = (self.position[q] - self.position[q-1]) / self.dt
    
    # ------------------ 3. COMPUTE SPEED ------------------
    def get_speed_profile(self):
        """
        Return speed at each time slot
        """
        speeds = np.linalg.norm(self.velocity, axis=1)
        return speeds
    
    # ------------------ 4. ADJUST HEIGHT (DYNAMIC BEHAVIOR) ------------------
    def adjust_height(self, min_h=80, max_h=150):
        """
        Slightly vary UAV altitude for realism
        """
        variation = np.random.uniform(-5, 5, size=self.Q)
        self.position[:, 2] = np.clip(self.position[:, 2] + variation, min_h, max_h)
    
    # ------------------ 5. DEBUG INFO ------------------
    def print_state(self):
        print("🛩️ UAV State:")
        print(f"   Time Slots      : {self.Q}")
        print(f"   Total Time      : {self.T}")
        print(f"   Avg Height      : {np.mean(self.position[:,2]):.2f}")
        print(f"   Avg Power Alpha : {np.mean(self.alpha):.2f}")