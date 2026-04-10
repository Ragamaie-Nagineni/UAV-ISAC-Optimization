import matplotlib.pyplot as plt

def plot_trajectory(history, env):
    plt.figure()

    plt.scatter(env.nodes[:,0], env.nodes[:,1], c='red', label='Nodes')

    # plot all iterations
    for pos in history:
        plt.plot(pos[:,0], pos[:,1], alpha=0.3)

    # highlight final path
    plt.plot(history[-1][:,0], history[-1][:,1], 'blue', linewidth=2, label='Final Path')

    plt.legend()
    plt.title("UAV Trajectory Evolution")
    plt.show()