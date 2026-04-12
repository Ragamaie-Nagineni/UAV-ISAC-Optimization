import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# ------------------ 1. TRAJECTORY EVOLUTION ------------------
def plot_trajectory(history, env):
    plt.figure(figsize=(6,5))

    plt.scatter(env.nodes[:,0], env.nodes[:,1], c='red', label='Nodes')

    # evolution
    for pos in history:
        plt.plot(pos[:,0], pos[:,1], alpha=0.2)

    # final path
    plt.plot(history[-1][:,0], history[-1][:,1], 'blue', linewidth=2, label='Final Path')

    plt.legend()
    plt.title("UAV Trajectory Evolution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.tight_layout()
    plt.show()


# ------------------ 2. BEFORE vs AFTER ------------------
def plot_before_after(initial_pos, final_pos, env):
    plt.figure(figsize=(6,5))

    plt.scatter(env.nodes[:,0], env.nodes[:,1], c='red', label='Nodes')

    plt.plot(initial_pos[:,0], initial_pos[:,1], 'orange', label='Initial')
    plt.plot(final_pos[:,0], final_pos[:,1], 'blue', linewidth=2, label='Optimized')

    plt.legend()
    plt.title("Before vs After Optimization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.tight_layout()
    plt.show()


# ------------------ 3. RATE vs ITERATION ------------------
def plot_rate(rates):
    plt.figure(figsize=(6,5))

    plt.plot(rates, marker='o')
    plt.title("Radar Estimation Rate vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Radar Rate")
    plt.grid()
    plt.tight_layout()

    plt.show()


# ------------------ 4. TASK SCHEDULING ------------------
def plot_schedule(schedule_list):
    indices = []

    for task, k in schedule_list:
        #indices.append(k if task == "ISAC" else -1)
        if task == "ISAC":
          indices.append(1)
        else:
          indices.append(0)

    plt.figure(figsize=(6,5))
    plt.bar(range(len(indices)), indices)

    plt.title("Task Scheduling over Time")
    plt.xlabel("Time Slot")
    plt.ylabel("Node Index (-1 = Upload)")
    plt.grid(axis='y')

    plt.tight_layout()
    plt.show()


# ------------------ 5. POWER ALLOCATION ------------------
def plot_power(uav):
    plt.figure(figsize=(6,5))

    plt.plot(uav.alpha)
    plt.title("Power Allocation (Alpha) over Time")
    plt.xlabel("Time Slot")
    plt.ylabel("Alpha (Communication Ratio)")
    plt.grid()

    plt.tight_layout()
    plt.show()


# ------------------ 6. UAV SPEED PROFILE ------------------
def plot_speed(uav):
    speeds = uav.get_speed_profile()

    plt.figure(figsize=(6,5))
    plt.plot(speeds)

    plt.title("UAV Speed over Time")
    plt.xlabel("Time Slot")
    plt.ylabel("Speed")
    plt.grid()

    plt.tight_layout()
    plt.show()


# ------------------ 7. 3D TRAJECTORY (BIG UPGRADE) ------------------
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_trajectory(history, env):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot nodes
    ax.scatter(env.nodes[:,0], env.nodes[:,1], np.zeros(len(env.nodes)),
               c='red', label='Nodes')

    # 🔥 plot all iterations (faded)
    for pos in history:
        ax.plot(pos[:,0], pos[:,1], pos[:,2], alpha=0.2)

    # 🔥 highlight final trajectory
    final = history[-1]
    ax.plot(final[:,0], final[:,1], final[:,2], linewidth=3, label='Final Path')

    ax.set_title("3D UAV Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")

    ax.legend()
    plt.show()


# ------------------ 8. ALL-IN-ONE DASHBOARD ------------------
def plot_all(history, env, initial_pos, final_pos, rates, schedule_list, uav):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Trajectory evolution
    axs[0,0].scatter(env.nodes[:,0], env.nodes[:,1], c='red')
    for pos in history:
        axs[0,0].plot(pos[:,0], pos[:,1], alpha=0.2)
    axs[0,0].plot(history[-1][:,0], history[-1][:,1], 'blue')
    axs[0,0].set_title("Trajectory Evolution")

    # 2. Before vs After
    axs[0,1].scatter(env.nodes[:,0], env.nodes[:,1], c='red')
    axs[0,1].plot(initial_pos[:,0], initial_pos[:,1], 'orange')
    axs[0,1].plot(final_pos[:,0], final_pos[:,1], 'blue')
    axs[0,1].set_title("Before vs After")

    # 3. Rate
    axs[0,2].plot(rates)
    axs[0,2].set_title("Radar Rate")

    # 4. Scheduling
    indices = [k if task=="ISAC" else -1 for task,k in schedule_list]
    axs[1,0].bar(range(len(indices)), indices)
    axs[1,0].set_title("Scheduling")

    # 5. Power
    axs[1,1].plot(uav.alpha)
    axs[1,1].set_title("Power Allocation")

    # 6. Speed
    speeds = uav.get_speed_profile()
    axs[1,2].plot(speeds)
    axs[1,2].set_title("Speed")

    plt.tight_layout()
    plt.show()