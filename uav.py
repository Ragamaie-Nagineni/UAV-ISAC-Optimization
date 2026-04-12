import numpy as np


class UAV:
    """
    UAV state for the ISAC optimisation (Section II of paper).

    Trajectory is time-discretised into Q slots of length dt = T/Q.
    The UAV operates in 3-D; altitude is optimised together with the
    horizontal path.
    """

    # ---- physical limits (Table II) ----
    V_XY_MAX  = 40.0   # m/s  max horizontal speed
    V_Z_MAX   = 20.0   # m/s  max vertical speed
    A_XY_MAX  = 5.0    # m/s² max horizontal acceleration
    A_Z_MAX   = 5.0    # m/s² max vertical acceleration
    H_MIN     = 100.0  # m    min flight altitude
    H_MAX     = 200.0  # m    max flight altitude
    THETA     = np.deg2rad(30)   # max radar detection half-angle (30°)
    P_AVG     = 1.0    # W    average transmit power budget

    def __init__(self, Q=200, T=100.0):
        self.Q  = Q          # number of time slots
        self.T  = T          # total flight cycle time (s)
        self.dt = T / Q      # slot duration (s)

        # ---- trajectory: (Q, 3) arrays ----
        self.position = np.zeros((Q, 3))   # [x, y, z]
        self.velocity = np.zeros((Q, 3))   # [vx, vy, vz]

        # ---- resource variables ----
        self.Pt    = np.ones(Q) * self.P_AVG   # total transmit power (W)
        self.alpha = np.ones(Q) * 0.5          # power split (comm fraction)

    # ------------------------------------------------------------------
    # Initialisation: circular path at mid altitude (eq. 59-61 of paper)
    # ------------------------------------------------------------------
    def initialize_trajectory(self, env):
        """
        Circular initial trajectory centred on the data-collection centre.
        Radius = mean of (max, min) distances from centre to nodes.
        Height = (H_MIN + H_MAX) / 2
        """
        center = env.data_center
        dists  = np.linalg.norm(env.nodes - center, axis=1)
        r      = (dists.max() + dists.min()) / 2.0
        r      = np.clip(r, 100, 550)
        H_init = (self.H_MIN + self.H_MAX) / 2.0

        # Starting angle aligned with node 0 (eq. 61 of paper)
        n0 = env.nodes[0]
        phi0 = np.arctan2(n0[1] - center[1], n0[0] - center[0])

        for q in range(self.Q):
            gamma = 2 * np.pi * q / self.Q + phi0
            self.position[q] = [
                center[0] + r * np.cos(gamma),
                center[1] + r * np.sin(gamma),
                H_init
            ]

        # Compute corresponding velocities
        omega = 2 * np.pi / self.T          # angular rate (rad/s)
        for q in range(self.Q):
            gamma = 2 * np.pi * q / self.Q + phi0
            self.velocity[q] = [
                -r * omega * np.sin(gamma),
                 r * omega * np.cos(gamma),
                 0.0
            ]

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def Pcom(self):
        """Communication power at each slot."""
        return self.alpha * self.Pt

    @property
    def Prad(self):
        """Radar (sensing) power at each slot."""
        return (1.0 - self.alpha) * self.Pt

    def get_speed_profile(self):
        """Speed magnitude at each time slot."""
        return np.linalg.norm(self.velocity, axis=1)

    # ------------------------------------------------------------------
    def print_state(self):
        print("🛩️  UAV State:")
        print(f"   Time Slots      : {self.Q}")
        print(f"   Total Time      : {self.T} s")
        print(f"   Slot Duration   : {self.dt:.3f} s")
        print(f"   Avg Height      : {np.mean(self.position[:, 2]):.2f} m")
        print(f"   Avg Pt          : {np.mean(self.Pt):.3f} W")
        print(f"   Avg alpha       : {np.mean(self.alpha):.3f}")
