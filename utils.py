import numpy as np

# ------------------ 1. DISTANCE ------------------
def distance(uav_pos, node_pos):
    """
    Euclidean distance between UAV and node
    """
    return np.linalg.norm(uav_pos - node_pos)


# ------------------ 2. COMMUNICATION CHANNEL ------------------
def h_com(d):
    """
    Communication channel gain ~ 1/d^2
    """
    return 1 / (d**2 + 1e-6)


# ------------------ 3. RADAR CHANNEL ------------------
def h_rad(d):
    """
    Radar channel gain ~ 1/d^4 (round-trip loss)
    """
    return 1 / (d**4 + 1e-6)


# ------------------ 4. RATE FUNCTION ------------------
def rate(sinr):
    """
    Shannon capacity formula
    """
    return np.log2(1 + sinr)


# ------------------ 5. NEW: SINR HELPER ------------------
def compute_sinr(signal, interference, noise=1e-9):
    """
    Generic SINR calculation
    """
    return signal / (interference + noise)


# ------------------ 6. NEW: NORMALIZE VECTOR ------------------
def normalize(vec):
    """
    Normalize a vector safely
    """
    return vec / (np.linalg.norm(vec) + 1e-6)


# ------------------ 7. NEW: DEBUG UTILITY ------------------
def print_stats(name, values):
    """
    Print statistics for debugging / console output
    """
    print(f"{name}:")
    print(f"   Min : {np.min(values):.3f}")
    print(f"   Max : {np.max(values):.3f}")
    print(f"   Avg : {np.mean(values):.3f}")