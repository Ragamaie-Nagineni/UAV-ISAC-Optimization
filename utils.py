import numpy as np

def distance(uav_pos, node_pos):
    return np.linalg.norm(uav_pos - node_pos)

def h_com(d):
    return 1 / (d**2 + 1e-6)

def h_rad(d):
    return 1 / (d**4 + 1e-6)

def rate(sinr):
    return np.log2(1 + sinr)