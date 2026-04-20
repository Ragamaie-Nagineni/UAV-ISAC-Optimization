"""
optimization.py
===============
Three-layer iterative optimisation from Section III of:

  Liu et al., "UAV Assisted Integrated Sensing and Communications for IoT:
  3D Trajectory Optimisation and Resource Allocation," IEEE TWC 2024.

Uses only numpy + scipy (no CVXPY required).

Sub-problem 1 – Task scheduling  (greedy, constraint-aware)
Sub-problem 2 – Power allocation (alternating SCA via scipy L-BFGS-B)
Sub-problem 3 – 3-D trajectory   (node-visiting with velocity smoothing)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter1d

from utils import (
    h_com, h_rad, distance_3d,
    sinr_com, sinr_rad, radar_rate, comm_rate, upload_rate,
    N0, B, beta_com, beta_rad
)


# ─────────────────────────────────────────────────────────────────────────────
# Channel gains
# ─────────────────────────────────────────────────────────────────────────────

def _channel_gains(uav, env):
    Q, K   = uav.Q, env.num_nodes
    hk_com = np.zeros((Q, K))
    hk_rad = np.zeros((Q, K))
    hc     = np.zeros(Q)
    dk     = np.zeros((Q, K))
    dc     = np.zeros(Q)

    for q in range(Q):
        pos_u = uav.position[q]
        dc[q] = distance_3d(pos_u, env.center_pos3d())
        hc[q] = h_com(dc[q])
        for k in range(K):
            d             = distance_3d(pos_u, env.node_pos3d(k))
            dk[q, k]      = d
            hk_com[q, k]  = h_com(d)
            hk_rad[q, k]  = h_rad(d)

    return hk_com, hk_rad, hc, dk, dc


def _all_rates(uav, hk_com, hk_rad, hc):
    Q, K    = uav.Q, hk_com.shape[1]
    Rrad    = np.zeros((Q, K))
    Rcom    = np.zeros((Q, K))
    Rc      = np.zeros(Q)
    for q in range(Q):
        Pc, Pr, Pt = uav.Pcom[q], uav.Prad[q], uav.Pt[q]
        Rc[q] = upload_rate(Pt, hc[q])
        for k in range(K):
            Rrad[q, k] = radar_rate(sinr_rad(Pr, Pc, hk_rad[q, k]))
            Rcom[q, k] = comm_rate(sinr_com(Pc, Pr, hk_com[q, k]))
    return Rrad, Rcom, Rc


# ─────────────────────────────────────────────────────────────────────────────
# Sub-problem 1: Task scheduling
# ─────────────────────────────────────────────────────────────────────────────

def solve_scheduling(uav, env, hk_com, hk_rad, hc, **_):
    """
    Greedy scheduling: each slot assigns ISAC to the best reachable node.
    A node is reachable if it is within the radar detection cone AND
    the comm-rate constraint (Rrad <= Rcom) is satisfied.
    Remaining slots assigned to data-upload.
    Global upload-capacity constraint enforced afterwards.
    """
    Q, K  = uav.Q, env.num_nodes
    omega = np.zeros((Q, K))
    b     = np.zeros(Q)
    tan2  = np.tan(uav.THETA) ** 2

    Rrad_all, Rcom_all, Rc_all = _all_rates(uav, hk_com, hk_rad, hc)

    total_isac_rate   = 0.0
    total_upload_rate = 0.0

    for q in range(Q):
        Hu     = uav.position[q, 2]
        xu, yu = uav.position[q, 0], uav.position[q, 1]

        candidates = []
        for k in range(K):
            xk, yk   = env.nodes[k]
            horiz2   = (xu - xk)**2 + (yu - yk)**2
            in_range = horiz2 <= Hu**2 * tan2
            feasible = Rrad_all[q, k] <= Rcom_all[q, k] + 1e-9
            if in_range and feasible:
                candidates.append((Rrad_all[q, k], k))

        if candidates:
            best_k = max(candidates, key=lambda x: x[0])[1]
            omega[q, best_k] = 1
            total_isac_rate += Rrad_all[q, best_k]
        else:
            b[q] = 1
            total_upload_rate += Rc_all[q]

    # Enforce global upload capacity (eq. 23g)
    if total_isac_rate > total_upload_rate + 1e-9:
        isac_slots = [(q, int(np.argmax(omega[q])))
                      for q in range(Q) if omega[q].sum() > 0.5]
        isac_slots.sort(key=lambda x: Rrad_all[x[0], x[1]])
        for q, k in isac_slots:
            if total_isac_rate <= total_upload_rate + 1e-9:
                break
            total_isac_rate  -= Rrad_all[q, k]
            omega[q, k]       = 0
            b[q]              = 1
            total_upload_rate += Rc_all[q]

    return omega, b, Rrad_all, Rcom_all, Rc_all


# ─────────────────────────────────────────────────────────────────────────────
# Sub-problem 2: Power allocation
# ─────────────────────────────────────────────────────────────────────────────

def solve_power(uav, env, omega, b, hk_com, hk_rad, hc, tol=1e-5, max_iter=60):
    """
    Alternating SCA over alpha and Pt using scipy L-BFGS-B.
    Updates uav.alpha and uav.Pt in-place.
    """
    Q, K      = uav.Q, env.num_nodes
    alpha_cur = uav.alpha.copy()
    Pt_cur    = uav.Pt.copy()

    for _ in range(max_iter):

        # ── Optimise alpha (Pt fixed) ──────────────────────────────────────
        def neg_rate_alpha(a_vec):
            val = 0.0
            for q in range(Q):
                a, Pt = a_vec[q], Pt_cur[q]
                Pc, Pr = a * Pt, (1 - a) * Pt
                for k in range(K):
                    if omega[q, k] < 0.5:
                        continue
                    rr      = radar_rate(sinr_rad(Pr, Pc, hk_rad[q, k]))
                    rc      = comm_rate(sinr_com(Pc, Pr, hk_com[q, k]))
                    penalty = max(0.0, rr - rc) * 1e3
                    val    += rr - penalty
            return -val

        res_a     = minimize(neg_rate_alpha, alpha_cur,
                             method="L-BFGS-B",
                             bounds=[(0.05, 0.95)] * Q,
                             options={"maxiter": 200, "ftol": 1e-10})
        alpha_new = np.clip(res_a.x, 0.05, 0.95)

        # ── Optimise Pt (alpha fixed) ──────────────────────────────────────
        def neg_rate_Pt(pt_vec):
            val = 0.0
            for q in range(Q):
                a, Pt = alpha_new[q], pt_vec[q]
                Pc, Pr = a * Pt, (1 - a) * Pt
                for k in range(K):
                    if omega[q, k] < 0.5:
                        continue
                    rr      = radar_rate(sinr_rad(Pr, Pc, hk_rad[q, k]))
                    rc      = comm_rate(sinr_com(Pc, Pr, hk_com[q, k]))
                    penalty = max(0.0, rr - rc) * 1e3
                    val    += rr - penalty
            avg_excess = max(0.0, np.mean(pt_vec) - uav.P_AVG) * 1e4
            return -(val - avg_excess)

        res_p  = minimize(neg_rate_Pt, Pt_cur,
                          method="L-BFGS-B",
                          bounds=[(0.0, 5 * uav.P_AVG)] * Q,
                          options={"maxiter": 200, "ftol": 1e-10})
        Pt_new = np.clip(res_p.x, 0.0, 5 * uav.P_AVG)
        # Rescale so average power equals P_AVG
        Pt_new = Pt_new * uav.P_AVG / (np.mean(Pt_new) + 1e-12)

        converged = (np.max(np.abs(alpha_new - alpha_cur)) < tol and
                     np.max(np.abs(Pt_new    - Pt_cur))   < tol)
        alpha_cur, Pt_cur = alpha_new, Pt_new
        if converged:
            break

    uav.alpha = alpha_cur
    uav.Pt    = Pt_cur


# ─────────────────────────────────────────────────────────────────────────────
# Sub-problem 3: 3-D trajectory optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _smooth_trajectory(pos_targets, uav, env, omega, b):
    """
    Build a kinematically feasible trajectory that:
      - Flies directly over each ISAC-scheduled node (horiz) at low altitude
      - Flies toward data-centre for upload slots at mid altitude
      - Satisfies max speed / acceleration / altitude bounds
      - Enforces periodic boundary (start == end)

    Strategy
    --------
    1. For each slot set a 'desired' (x,y,z) waypoint.
    2. Smooth horizontally with a sliding-window velocity limiter.
    3. Clip altitudes.
    4. Enforce periodicity by blending start/end.
    """
    Q, K   = uav.Q, env.num_nodes
    dt     = uav.dt
    pos    = pos_targets.copy()

    # ── Desired waypoints per slot ──────────────────────────────────────────
    desired = np.zeros((Q, 3))
    for q in range(Q):
        if omega[q].sum() > 0.5:                      # ISAC slot
            k = int(np.argmax(omega[q]))
            desired[q, 0] = env.nodes[k, 0]           # fly directly over node
            desired[q, 1] = env.nodes[k, 1]
            desired[q, 2] = uav.H_MIN + 10            # low altitude → better gain
        else:                                          # upload slot
            desired[q, 0] = env.data_center[0]
            desired[q, 1] = env.data_center[1]
            desired[q, 2] = (uav.H_MIN + uav.H_MAX) / 2

    # ── Smooth desired toward feasible with velocity limits ─────────────────
    # Use a weighted blend between current and desired, then enforce speed cap
    new_pos = pos.copy()
    BLEND   = 0.30   # fraction to move toward desired per outer call

    for q in range(Q):
        new_pos[q] = pos[q] + BLEND * (desired[q] - pos[q])

    # Enforce speed constraints slot-by-slot (forward pass)
    max_step_xy = uav.V_XY_MAX * dt
    max_step_z  = uav.V_Z_MAX  * dt

    for q in range(Q - 1):
        dxy  = new_pos[q+1, :2] - new_pos[q, :2]
        dz   = new_pos[q+1, 2]  - new_pos[q, 2]
        nxy  = np.linalg.norm(dxy)
        if nxy > max_step_xy:
            new_pos[q+1, :2] = new_pos[q, :2] + dxy / nxy * max_step_xy
        if abs(dz) > max_step_z:
            new_pos[q+1, 2]  = new_pos[q, 2] + np.sign(dz) * max_step_z

    # Altitude bounds
    new_pos[:, 2] = np.clip(new_pos[:, 2], uav.H_MIN, uav.H_MAX)

    # Enforce periodicity: linearly blend last slot back toward first slot
    new_pos[Q-1] = new_pos[0]

    return new_pos


def solve_trajectory(uav, env, omega, b, tol=1e-4, max_iter=30):
    """
    Iteratively move the UAV trajectory toward node-visiting waypoints,
    re-evaluating the radar rate at each step and keeping the improvement.
    Updates uav.position and uav.velocity in-place.
    """
    Q  = uav.Q
    dt = uav.dt

    pos      = uav.position.copy()
    prev_obj = _trajectory_rate(pos, uav, env, omega)

    for it in range(max_iter):
        pos_new = _smooth_trajectory(pos, uav, env, omega, b)
        new_obj = _trajectory_rate(pos_new, uav, env, omega)

        if new_obj >= prev_obj - 1e-9:
            pos      = pos_new
            prev_obj = new_obj
        # Always continue for max_iter; stop early if change is tiny
        if it > 2 and abs(new_obj - prev_obj) < tol:
            break

    uav.position = pos.copy()
    for q in range(Q - 1):
        uav.velocity[q] = (uav.position[q+1] - uav.position[q]) / dt
    uav.velocity[Q-1] = uav.velocity[Q-2]


def _trajectory_rate(pos, uav, env, omega):
    """Total radar rate for a candidate position array (without modifying uav)."""
    total = 0.0
    Q, K  = uav.Q, env.num_nodes
    for q in range(Q):
        for k in range(K):
            if omega[q, k] < 0.5:
                continue
            d   = distance_3d(pos[q], env.node_pos3d(k))
            sr  = sinr_rad(uav.Prad[q], uav.Pcom[q], h_rad(d))
            total += radar_rate(sr)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Utility: total radar estimation rate
# ─────────────────────────────────────────────────────────────────────────────

def compute_total_radar_rate(uav, env, omega):
    return _trajectory_rate(uav.position, uav, env, omega)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 2: three-layer iterative optimisation
# ─────────────────────────────────────────────────────────────────────────────

def three_layer_optimize(uav, env, max_outer=20, tol=1e-3, verbose=True):
    """
    Algorithm 2 from the paper:
      Repeat until convergence:
        1. Fix P, L  → optimise W  (scheduling)
        2. Fix W, L  → optimise P  (power)
        3. Fix W, P  → optimise L  (trajectory)

    Returns
    -------
    rates      : list[float]       radar rate after each outer iteration
    omega_hist : list[np.ndarray]  scheduling matrices
    """
    rates      = []
    omega_hist = []
    prev_rate  = -np.inf

    for i in range(max_outer):
        # ── Channel gains (current trajectory) ────────────────────────────
        hk_com, hk_rad, hc, dk, dc = _channel_gains(uav, env)

        # ── Step 1: Task scheduling ───────────────────────────────────────
        omega, b, Rrad, Rcom, Rc = solve_scheduling(uav, env, hk_com, hk_rad, hc)

        # ── Step 2: Power allocation ──────────────────────────────────────
        solve_power(uav, env, omega, b, hk_com, hk_rad, hc)

        # ── Step 3: 3-D trajectory ────────────────────────────────────────
        solve_trajectory(uav, env, omega, b)

        # ── Evaluate ──────────────────────────────────────────────────────
        rate = compute_total_radar_rate(uav, env, omega)
        rates.append(rate)
        omega_hist.append(omega.copy())

        if verbose:
            isac_cnt   = int(omega.sum())
            upload_cnt = int(b.sum())
            print(f"  Iter {i+1:2d}  |  Radar Rate: {rate:9.4f} bps/Hz"
                  f"  |  ISAC slots: {isac_cnt:3d}  |  Upload slots: {upload_cnt:3d}")

        # ── Convergence ───────────────────────────────────────────────────
        if abs(rate - prev_rate) < tol and i > 1:
            if verbose:
                print(f"  ✅ Converged at iteration {i+1}")
            break
        prev_rate = rate

    return rates, omega_hist
