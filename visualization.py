import matplotlib.pyplot as plt

def plot_trajectory(uav, env):
    x = uav.position[:, 0]
    y = uav.position[:, 1]
    
    plt.scatter(env.nodes[:,0], env.nodes[:,1], c='red', label='Nodes')
    plt.plot(x, y, label='UAV Path')
    
    plt.legend()
    plt.title("UAV Trajectory")
    plt.show()