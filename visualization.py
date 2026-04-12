"""
visualization.py
================
All plotting functions for the UAV-ISAC simulation.
Each function saves a PNG file AND shows it (when a display is available).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all envs)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3d projection)
import os

OUTPUT_DIR = "output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _savefig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   💾  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Initial vs. Optimised 3-D trajectory
# ─────────────────────────────────────────────────────────────────────────────

def plot_3d_trajectory(initial_pos, final_pos, env, omega=None):
    """
    3-D plot mirroring Fig. 2(a) of the paper.
    Orange  = initial circular trajectory
    Blue    = optimised trajectory
    Markers = ISAC segments (orange dots) vs Upload segments (blue dots)
    """
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # Node positions
    ax.scatter(env.nodes[:, 0], env.nodes[:, 1],
               np.zeros(env.num_nodes),
               c="red", s=60, marker="^", zorder=5, label="IoT Nodes")
    # Data centre
    ax.scatter(*env.data_center, 0,
               c="black", s=100, marker="*", zorder=6, label="Data Centre")

    # Initial trajectory
    ax.plot(initial_pos[:, 0], initial_pos[:, 1], initial_pos[:, 2],
            color="orange", linewidth=1.2, alpha=0.6, label="Initial trajectory")

    # Optimised trajectory – colour by task
    if omega is not None:
        Q = final_pos.shape[0]
        K = omega.shape[1]
        for q in range(Q - 1):
            isac_slot = any(omega[q, k] > 0.5 for k in range(K))
            col = "#1f77b4" if isac_slot else "#2ca02c"
            ax.plot(final_pos[q:q+2, 0], final_pos[q:q+2, 1], final_pos[q:q+2, 2],
                    color=col, linewidth=2)
        # Legend proxies
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], color="#1f77b4", linewidth=2, label="Optimised – ISAC"),
            Line2D([0], [0], color="#2ca02c", linewidth=2, label="Optimised – Upload"),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + legend_elems, fontsize=8)
    else:
        ax.plot(final_pos[:, 0], final_pos[:, 1], final_pos[:, 2],
                color="#1f77b4", linewidth=2, label="Optimised trajectory")
        ax.legend(fontsize=8)

    # Node index labels
    for k in range(env.num_nodes):
        ax.text(env.nodes[k, 0], env.nodes[k, 1], 5,
                str(k + 1), fontsize=7, color="darkred")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Altitude (m)")
    ax.set_title("Initial vs. Optimised 3-D UAV Trajectory")
    _savefig(fig, "3d_trajectory.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Top-view (2-D) trajectory with task colouring
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_view(initial_pos, final_pos, env, omega=None):
    """Top-view equivalent of Fig. 3 of the paper."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(env.nodes[:, 0], env.nodes[:, 1],
               c="red", s=60, marker="^", zorder=5, label="IoT Nodes")
    ax.scatter(*env.data_center, c="black", s=100, marker="*",
               zorder=6, label="Data Centre")
    for k in range(env.num_nodes):
        ax.annotate(str(k + 1), env.nodes[k], fontsize=8,
                    color="darkred", xytext=(5, 5), textcoords="offset points")

    ax.plot(initial_pos[:, 0], initial_pos[:, 1],
            color="orange", linewidth=1.2, alpha=0.6, label="Initial trajectory")

    if omega is not None:
        Q = final_pos.shape[0]
        K = omega.shape[1]
        for q in range(Q - 1):
            isac_slot = any(omega[q, k] > 0.5 for k in range(K))
            col = "#1f77b4" if isac_slot else "#2ca02c"
            ax.plot(final_pos[q:q+2, 0], final_pos[q:q+2, 1],
                    color=col, linewidth=1.8)
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], color="#1f77b4", linewidth=2, label="Optimised – ISAC"),
            Line2D([0], [0], color="#2ca02c", linewidth=2, label="Optimised – Upload"),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + legend_elems, fontsize=9)
    else:
        ax.plot(final_pos[:, 0], final_pos[:, 1],
                color="#1f77b4", linewidth=2, label="Optimised trajectory")
        ax.legend(fontsize=9)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Top View – Initial vs. Optimised Trajectory")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    _savefig(fig, "top_view_trajectory.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Radar estimation rate vs. outer iterations (Fig. 7 / Fig. 12 equivalent)
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(rates, label="3-D opt."):
    """Convergence curve of the three-layer algorithm."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(rates) + 1), rates,
            marker="o", linewidth=2, label=label)
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Sum Radar Estimation Rate (bps/Hz)")
    ax.set_title("Convergence of Three-Layer Optimisation")
    ax.legend(); ax.grid(True, alpha=0.4)
    _savefig(fig, "convergence_rate.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Task scheduling bar chart (Fig. 4 equivalent)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scheduling(omega, b, env):
    """
    Bar chart: which node (or data centre) the UAV serves each time slot.
    """
    Q = omega.shape[0]
    K = omega.shape[1]
    schedule_idx = np.full(Q, -1, dtype=int)   # -1 = idle (shouldn't occur)

    for q in range(Q):
        if b[q] > 0.5:
            schedule_idx[q] = 0                # 0 → data centre
        else:
            for k in range(K):
                if omega[q, k] > 0.5:
                    schedule_idx[q] = k + 1    # 1…K → node k+1

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = plt.cm.tab20(np.linspace(0, 1, K + 1))
    bar_colors = [colors[s] if s >= 0 else "grey" for s in schedule_idx]
    ax.bar(range(Q), np.ones(Q), color=bar_colors, width=1.0)

    # Custom legend
    from matplotlib.patches import Patch
    patches = [Patch(color=colors[0], label="Data Centre")]
    for k in range(K):
        patches.append(Patch(color=colors[k + 1], label=f"Node {k+1}"))
    ax.legend(handles=patches, fontsize=7, ncol=4,
              loc="upper right", framealpha=0.8)

    ax.set_xlabel("Time Slot Index")
    ax.set_ylabel("Assigned Target")
    ax.set_title("UAV Task Scheduling per Time Slot")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.3)
    _savefig(fig, "task_scheduling.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. UAV altitude profile
# ─────────────────────────────────────────────────────────────────────────────

def plot_altitude(initial_pos, final_pos):
    """Altitude vs. time slot for initial and optimised trajectory."""
    Q = final_pos.shape[0]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(Q), initial_pos[:, 2],
            color="orange", linewidth=1.5, alpha=0.7, label="Initial")
    ax.plot(range(Q), final_pos[:, 2],
            color="#1f77b4", linewidth=2, label="Optimised")
    ax.axhline(100, color="grey", linestyle="--", linewidth=0.8, label="H_min")
    ax.axhline(200, color="grey", linestyle=":",  linewidth=0.8, label="H_max")
    ax.set_xlabel("Time Slot Index")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("UAV Flight Altitude Profile")
    ax.legend(); ax.grid(True, alpha=0.4)
    _savefig(fig, "altitude_profile.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. UAV speed profile (Fig. 6 equivalent)
# ─────────────────────────────────────────────────────────────────────────────

def plot_speed(initial_vel, final_vel):
    """Speed magnitude vs. time slot."""
    Q = final_vel.shape[0]
    fig, ax = plt.subplots(figsize=(9, 4))

    init_speed  = np.linalg.norm(initial_vel, axis=1)
    final_speed = np.linalg.norm(final_vel,   axis=1)
    h_speed     = np.linalg.norm(final_vel[:, :2], axis=1)
    v_speed     = np.abs(final_vel[:, 2])

    ax.plot(range(Q), init_speed,  color="orange", linewidth=1.2,
            alpha=0.6, label="Initial speed")
    ax.plot(range(Q), final_speed, color="#1f77b4", linewidth=2,
            label="Optimised total speed")
    ax.plot(range(Q), h_speed,     color="#2ca02c", linewidth=1.2,
            linestyle="--", label="Horizontal speed")
    ax.plot(range(Q), v_speed,     color="#d62728", linewidth=1.2,
            linestyle=":",  label="Vertical speed")

    ax.set_xlabel("Time Slot Index")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("UAV Speed Profile")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    _savefig(fig, "speed_profile.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Power allocation (Fig. 8 equivalent)
# ─────────────────────────────────────────────────────────────────────────────

def plot_power(uav, omega, b):
    """
    Stacked bar: communication power vs. radar power per time slot.
    Upload slots shown in green.
    """
    Q = uav.Q
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    slots = np.arange(Q)

    # Top: power split
    axes[0].bar(slots, uav.Pcom, label="P_com (α·Pt)", color="#1f77b4", width=1.0)
    axes[0].bar(slots, uav.Prad, bottom=uav.Pcom,
                label="P_rad ((1-α)·Pt)", color="#d62728", width=1.0)
    axes[0].set_ylabel("Power (W)")
    axes[0].set_title("Transmit Power Allocation per Time Slot")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.4)

    # Bottom: alpha
    axes[1].plot(slots, uav.alpha, color="#ff7f0e", linewidth=1.5)
    axes[1].axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("α (comm. fraction)")
    axes[1].set_xlabel("Time Slot Index")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Power Split Factor α")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    _savefig(fig, "power_allocation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. All-in-one dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(initial_pos, final_pos, uav, env, rates, omega, b):
    """6-panel dashboard summarising the full optimisation."""
    Q = uav.Q
    K = env.num_nodes
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("UAV-ISAC Optimisation Dashboard", fontsize=15, fontweight="bold")

    # ── Panel 1: Top-view trajectory ────────────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(env.nodes[:, 0], env.nodes[:, 1],
               c="red", s=50, marker="^", zorder=5, label="Nodes")
    ax.scatter(*env.data_center, c="black", s=100, marker="*",
               zorder=6, label="Centre")
    for k in range(K):
        ax.annotate(str(k+1), env.nodes[k], fontsize=7, color="darkred",
                    xytext=(3, 3), textcoords="offset points")
    ax.plot(initial_pos[:, 0], initial_pos[:, 1],
            color="orange", linewidth=1, alpha=0.6, label="Initial")
    # colour optimised by task
    for q in range(Q - 1):
        isac_slot = any(omega[q, k] > 0.5 for k in range(K))
        col = "#1f77b4" if isac_slot else "#2ca02c"
        ax.plot(final_pos[q:q+2, 0], final_pos[q:q+2, 1],
                color=col, linewidth=1.5)
    ax.set_title("Top-View Trajectory")
    ax.legend(fontsize=7); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    # ── Panel 2: Convergence ────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(range(1, len(rates)+1), rates, marker="o", linewidth=2, color="#9467bd")
    ax.set_title("Convergence")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Radar Rate (bps/Hz)")
    ax.grid(True, alpha=0.4)

    # ── Panel 3: Altitude ────────────────────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(range(Q), initial_pos[:, 2], color="orange", linewidth=1, alpha=0.6)
    ax.plot(range(Q), final_pos[:, 2], color="#1f77b4", linewidth=2)
    ax.axhline(uav.H_MIN, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(uav.H_MAX, color="grey", linestyle=":",  linewidth=0.8)
    ax.set_title("Altitude Profile"); ax.set_xlabel("Slot"); ax.set_ylabel("m")
    ax.grid(True, alpha=0.4)

    # ── Panel 4: Scheduling ──────────────────────────────────────────────────
    ax = axes[1, 0]
    schedule_idx = np.full(Q, -1, dtype=int)
    for q in range(Q):
        if b[q] > 0.5:
            schedule_idx[q] = 0
        else:
            for k in range(K):
                if omega[q, k] > 0.5:
                    schedule_idx[q] = k + 1
    colors_tab = plt.cm.tab20(np.linspace(0, 1, K + 1))
    bar_c = [colors_tab[s] if s >= 0 else "grey" for s in schedule_idx]
    ax.bar(range(Q), np.ones(Q), color=bar_c, width=1.0)
    ax.set_title("Task Scheduling"); ax.set_xlabel("Slot"); ax.set_yticks([])

    # ── Panel 5: Power alpha ─────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(range(Q), uav.alpha, color="#ff7f0e", linewidth=1.5)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Power Split α"); ax.set_xlabel("Slot"); ax.set_ylabel("α")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.4)

    # ── Panel 6: Speed ───────────────────────────────────────────────────────
    ax = axes[1, 2]
    speed = np.linalg.norm(uav.velocity, axis=1)
    ax.plot(range(Q), speed, color="#17becf", linewidth=1.5)
    ax.set_title("UAV Speed"); ax.set_xlabel("Slot"); ax.set_ylabel("m/s")
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    _savefig(fig, "dashboard.png")
