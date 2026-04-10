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

def plot_before_after(initial_pos, final_pos, env):
    plt.figure()

    plt.scatter(env.nodes[:,0], env.nodes[:,1], c='red', label='Nodes')

    plt.plot(initial_pos[:,0], initial_pos[:,1], 'orange', label='Initial Trajectory')
    plt.plot(final_pos[:,0], final_pos[:,1], 'blue', linewidth=2, label='Optimized Trajectory')

    plt.legend()
    plt.title("Before vs After Optimization")
    plt.show()

def plot_rate(rates):
    plt.figure()

    plt.plot(rates, marker='o')
    plt.title("Radar Rate vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Radar Rate")
    plt.grid()

    plt.show()

def plot_schedule(schedule_list):
    indices = []

    for task, k in schedule_list:
        if task == "ISAC":
            indices.append(k)
        else:
            indices.append(-1)

    plt.figure()
    plt.bar(range(len(indices)), indices)
    plt.title("Task Scheduling")
    plt.xlabel("Time Slot")
    plt.ylabel("Node Index (-1 = Upload)")

    plt.show()