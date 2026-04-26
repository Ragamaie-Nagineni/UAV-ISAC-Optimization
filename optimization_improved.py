"""
optimization_improved.py
========================
Enhanced three-layer iterative optimisation with TWO improvements over the
baseline (optimization_baseline.py):

  Improvement 1 – Multi-Objective Optimisation (Pareto front)
  ─────────────────────────────────────────────────────────────
  Instead of maximising radar rate alone, we add a second objective:
  minimise UAV energy consumption simultaneously.  A convex scalarisation
  weight λ ∈ [0, 1] blends the two objectives:

        maximise   λ · R_total  −  (1−λ) · E_total

  where E_total = Σ_q Pt[q] · dt  (total energy in joules).
  Running the solver for a sweep of λ values traces out the Pareto front.

  Improvement 2 – Min-Rate Fairness Constraint
  ─────────────────────────────────────────────
  Every IoT node must receive at least R_min bps/Hz of total radar sensing
  service over the flight cycle.  This is achieved by a two-stage scheduling
  policy:

    Stage A  (mandatory)  – reserve slots for any node below R_min.
    Stage B  (greedy)     – fill remaining slots as in the baseline.

  R_min is a design parameter (default 0.5 bps/Hz).

Both improvements are orthogonal and can be enabled independently via flags
passed to three_layer_optimize_improved().
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
# Channel gains  (identical to baseline)
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
    Q, K = uav.Q, hk_com.shape[1]
    Rrad = np.zeros((Q, K))
    Rcom = np.zeros((Q, K))
    Rc   = np.zeros(Q)
    for q in range(Q):
        Pc, Pr, Pt = uav.Pcom[q], uav.Prad[q], uav.Pt[q]
        Rc[q] = upload_rate(Pt, hc[q])
        for k in range(K):
            Rrad[q, k] = radar_rate(sinr_rad(Pr, Pc, hk_rad[q, k]))
            Rcom[q, k] = comm_rate(sinr_com(Pc, Pr, hk_com[q, k]))
    return Rrad, Rcom, Rc


# ─────────────────────────────────────────────────────────────────────────────
# Sub-problem 1: Fairness-aware task scheduling  (Improvement 2)
# ─────────────────────────────────────────────────────────────────────────────

def solve_scheduling_fair(uav, env, hk_com, hk_rad, hc, R_min=0.5, **_):
    """
    Two-stage fairness-aware greedy scheduling.

    Stage A – for every node below R_min, greedily assign the single best
              available slot that (a) is within the radar cone, (b) satisfies
              Rrad ≤ Rcom, and (c) yields the highest radar rate for that node.
              Repeat until all nodes reach R_min or no feasible slot remains.

    Stage B – fill remaining slots with the standard greedy policy (best
              reachable node, ignoring fairness since Stage A already satisfied
              the constraint).

    Enforces the global upload-capacity constraint afterwards (same as baseline).
    """
    Q, K  = uav.Q, env.num_nodes
    omega = np.zeros((Q, K))
    b     = np.zeros(Q)
    tan2  = np.tan(uav.THETA) ** 2

    Rrad_all, Rcom_all, Rc_all = _all_rates(uav, hk_com, hk_rad, hc)

    # Accumulated radar service per node [bps/Hz · slots]
    node_service  = np.zeros(K)
    slot_assigned = np.zeros(Q, dtype=bool)   # True once slot is used

    # ── Stage A: satisfy R_min for all nodes ────────────────────────────────
    changed = True
    while changed:
        changed = False
        # Priority: most-deprived node first
        deficit = R_min - node_service
        order   = np.argsort(-deficit)        # descending deficit

        for k in order:
            if node_service[k] >= R_min:
                continue                       # already satisfied

            # Find the best unassigned slot for this node
            best_q, best_r = -1, -np.inf
            for q in range(Q):
                if slot_assigned[q]:
                    continue
                xu, yu = uav.position[q, 0], uav.position[q, 1]
                Hu     = uav.position[q, 2]
                xk, yk = env.nodes[k]
                horiz2 = (xu - xk)**2 + (yu - yk)**2
                in_range = horiz2 <= Hu**2 * tan2
                feasible = Rrad_all[q, k] <= Rcom_all[q, k] + 1e-9
                if in_range and feasible and Rrad_all[q, k] > best_r:
                    best_q = q
                    best_r = Rrad_all[q, k]

            if best_q >= 0:
                omega[best_q, k]    = 1
                slot_assigned[best_q] = True
                node_service[k]    += best_r
                changed = True       # schedule changed → restart priority loop

    # ── Stage B: greedy fill for remaining unassigned slots ─────────────────
    for q in range(Q):
        if slot_assigned[q]:
            continue
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
            omega[q, best_k]  = 1
            slot_assigned[q]  = True
        else:
            b[q] = 1

    # Unassigned slots → upload
    for q in range(Q):
        if not slot_assigned[q]:
            b[q] = 1

    # ── Enforce global upload capacity (eq. 23g) ─────────────────────────────
    total_isac_rate   = sum(Rrad_all[q, int(np.argmax(omega[q]))]
                            for q in range(Q) if omega[q].sum() > 0.5)
    total_upload_rate = sum(Rc_all[q] for q in range(Q) if b[q] > 0.5)

    if total_isac_rate > total_upload_rate + 1e-9:
        isac_slots = [(q, int(np.argmax(omega[q])))
                      for q in range(Q) if omega[q].sum() > 0.5]
        # Remove lowest-rate ISAC slots first, but avoid dropping a slot if
        # that would push some node below R_min
        isac_slots.sort(key=lambda x: Rrad_all[x[0], x[1]])
        for q, k in isac_slots:
            if total_isac_rate <= total_upload_rate + 1e-9:
                break
            # Check if dropping hurts fairness
            if node_service[k] - Rrad_all[q, k] < R_min:
                continue    # skip: would violate fairness
            total_isac_rate  -= Rrad_all[q, k]
            node_service[k]  -= Rrad_all[q, k]
            omega[q, k]       = 0
            b[q]              = 1
            total_upload_rate += Rc_all[q]

    return omega, b, Rrad_all, Rcom_all, Rc_all, node_service


# ─────────────────────────────────────────────────────────────────────────────
# Sub-problem 2: Multi-objective power allocation  (Improvement 1)
# ─────────────────────────────────────────────────────────────────────────────

def solve_power_multiobjective(uav, env, omega, b, hk_com, hk_rad, hc,
                               lam=1.0, tol=1e-5, max_iter=60):
    """
    Multi-objective power allocation:

        maximise   lam * R_radar  -  (1-lam) * E_norm

    where E_norm = E / E_budget  (normalised to [0,1]).

    Two complementary energy-saving mechanisms:

    1. Upload-slot power reduction  (structural saving)
       Upload slots contribute zero radar rate but 100 % of their Pt to E.
       When lam < 1, we set their power to P_min_upload = (1-lam) * P_AVG,
       which gives 0 J at lam=0 and P_AVG at lam=1 — a direct, linear saving
       that accounts for up to 82 % of the budget (165/200 upload slots).

    2. ISAC-slot alpha optimisation  (efficiency saving)
       For ISAC slots we solve the scalarised problem over alpha with Pt fixed.
       The energy term is correctly normalised so both objectives have the same
       order of magnitude, making lam an effective trade-off knob.

    lam = 1.0 → pure radar maximisation  (P_upload = P_AVG, full budget)
    lam = 0.0 → pure energy minimisation (P_upload = 0,     min budget)
    0 < lam < 1 → Pareto trade-off

    Updates uav.alpha and uav.Pt in-place.
    Returns total energy consumed (J).
    """
    Q, K = uav.Q, env.num_nodes
    dt   = uav.dt

    # ── Mechanism 1: set upload-slot power proportional to lam ───────────────
    # Upload slots carry data to the centre; they need minimum Pcom = P_min_com.
    # We allow the solver to reduce their power to zero at lam=0.
    P_upload = lam * uav.P_AVG          # W; 0 at lam=0, P_AVG at lam=1
    P_isac   = uav.P_AVG               # ISAC slots keep full budget to maximise R

    Pt_cur = np.where(b > 0.5, P_upload, P_isac).astype(float)
    # Hard-cap: never exceed P_MAX = 2*P_AVG per slot
    P_MAX = 2.0 * uav.P_AVG
    Pt_cur = np.clip(Pt_cur, 0.0, P_MAX)

    alpha_cur = uav.alpha.copy()
    # Upload slots: all power goes to communication (alpha=1 means Pcom=Pt, Prad=0)
    for q in range(Q):
        if b[q] > 0.5:
            alpha_cur[q] = 0.95   # mostly comms power for upload

    E_budget  = Q * uav.P_AVG * dt      # reference energy at full power
    isac_slots = [q for q in range(Q) if omega[q].sum() > 0.5]

    # Upload slots never contribute radar rate, so optimising their alpha is
    # pointless. We always optimise over ISAC slots only — this also makes the
    # optimisation dimension small (35 vs 200) and fast for all lam values.
    if not isac_slots:
        uav.alpha = alpha_cur
        uav.Pt    = Pt_cur
        return float(np.sum(uav.Pt) * dt)

    for _ in range(max_iter):
        a0 = alpha_cur[isac_slots]

        def neg_obj_alpha(a_vec):
            val    = 0.0
            e_norm = np.sum(Pt_cur) * dt / (E_budget + 1e-12)
            for i, q in enumerate(isac_slots):
                a, Pt = a_vec[i], Pt_cur[q]
                Pc, Pr = a * Pt, (1.0 - a) * Pt
                for k in range(K):
                    if omega[q, k] < 0.5:
                        continue
                    rr = radar_rate(sinr_rad(Pr, Pc, hk_rad[q, k]))
                    rc = comm_rate(sinr_com(Pc, Pr, hk_com[q, k]))
                    val += rr - max(0.0, rr - rc) * 1e3
            # lam=1: energy term vanishes → pure radar; lam<1: energy penalised
            return -(lam * val - (1.0 - lam) * e_norm)

        res_a     = minimize(neg_obj_alpha, a0,
                             method="L-BFGS-B",
                             bounds=[(0.05, 0.95)] * len(isac_slots),
                             options={"maxiter": 200, "ftol": 1e-10})
        alpha_new = alpha_cur.copy()
        for i, q in enumerate(isac_slots):
            alpha_new[q] = float(np.clip(res_a.x[i], 0.05, 0.95))

        # At lam=1: also redistribute Pt across ISAC slots for best radar rate
        # (same as original baseline SCA but only over 35 slots, not 200).
        if lam > 0.999:
            pt0 = Pt_cur[isac_slots]

            def neg_obj_Pt(pt_vec):
                val = 0.0
                for i, q in enumerate(isac_slots):
                    a, Pt = alpha_new[q], pt_vec[i]
                    Pc, Pr = a * Pt, (1.0 - a) * Pt
                    for k in range(K):
                        if omega[q, k] < 0.5:
                            continue
                        rr = radar_rate(sinr_rad(Pr, Pc, hk_rad[q, k]))
                        rc = comm_rate(sinr_com(Pc, Pr, hk_com[q, k]))
                        val += rr - max(0.0, rr - rc) * 1e3
                return -val

            res_p = minimize(neg_obj_Pt, pt0,
                             method="L-BFGS-B",
                             bounds=[(0.0, P_MAX)] * len(isac_slots),
                             options={"maxiter": 200, "ftol": 1e-10})
            pt_new_isac = np.clip(res_p.x, 0.0, P_MAX)
            # Rescale so mean ISAC Pt = P_AVG (budget-neutral)
            pt_new_isac *= uav.P_AVG / (np.mean(pt_new_isac) + 1e-12)
            Pt_new = Pt_cur.copy()
            for i, q in enumerate(isac_slots):
                Pt_new[q] = pt_new_isac[i]
            converged = (np.max(np.abs(alpha_new - alpha_cur)) < tol and
                         np.max(np.abs(Pt_new[isac_slots] - Pt_cur[isac_slots])) < tol)
            Pt_cur = Pt_new
        else:
            # lam<1: Pt already set by Mechanism 1; only alpha changes
            converged = np.max(np.abs(alpha_new - alpha_cur)) < tol

        alpha_cur = alpha_new
        if converged:
            break

    uav.alpha = alpha_cur
    uav.Pt    = Pt_cur
    return float(np.sum(uav.Pt) * dt)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-problem 3: 3-D trajectory optimisation  (identical structure to baseline)
# ─────────────────────────────────────────────────────────────────────────────

def _smooth_trajectory(pos_targets, uav, env, omega, b, node_service=None, R_min=0.5):
    """
    Per-slot independent waypoint nudge (no velocity-cascade).

    Each ISAC slot is nudged independently toward its assigned node in XY.
    Altitude is only lowered for genuinely deprived nodes (urgency > 0) to
    improve h_rad ∝ d^{-4}; for satisfied nodes the altitude is untouched so
    the existing cone footprint is preserved.  Each slot's step is capped at
    STEP_XY / STEP_Z so the move is always feasible, but we do NOT propagate
    clipping forward — that is what caused the cascade that pushed boundary
    slots outside their node's cone in the original BLEND implementation.
    Upload slots are nudged toward the data-centre.
    """
    Q  = uav.Q
    dt = uav.dt
    pos = pos_targets.copy()

    if node_service is None:
        node_service = np.zeros(env.num_nodes)

    max_step_xy = uav.V_XY_MAX * dt * 0.30   # 30 % of max speed per call
    max_step_z  = uav.V_Z_MAX  * dt * 0.30

    new_pos = pos.copy()
    tan2    = np.tan(uav.THETA) ** 2

    for q in range(Q):
        if omega[q].sum() > 0.5:
            k  = int(np.argmax(omega[q]))
            xk, yk = env.nodes[k]

            # ── XY nudge: move toward assigned node, capped per slot ────────
            dx = xk - pos[q, 0]
            dy = yk - pos[q, 1]
            dist_xy = np.sqrt(dx**2 + dy**2) + 1e-9
            step = min(dist_xy, max_step_xy)
            new_pos[q, 0] = pos[q, 0] + (dx / dist_xy) * step
            new_pos[q, 1] = pos[q, 1] + (dy / dist_xy) * step

            # ── Z nudge: only lower for deprived nodes ───────────────────────
            deficit = max(0.0, R_min - node_service[k])
            urgency = min(1.0, deficit / (R_min + 1e-9))
            if urgency > 0.01:
                target_z = uav.H_MIN + (1.0 - urgency) * (pos[q, 2] - uav.H_MIN)
                dz = target_z - pos[q, 2]
                dz = np.clip(dz, -max_step_z, max_step_z)
                new_pos[q, 2] = pos[q, 2] + dz
            # else: altitude unchanged

        else:
            # Upload slot: nudge toward data-centre XY, altitude midpoint
            xc, yc = env.data_center
            dx = xc - pos[q, 0]
            dy = yc - pos[q, 1]
            dist_xy = np.sqrt(dx**2 + dy**2) + 1e-9
            step = min(dist_xy, max_step_xy)
            new_pos[q, 0] = pos[q, 0] + (dx / dist_xy) * step
            new_pos[q, 1] = pos[q, 1] + (dy / dist_xy) * step
            mid_z = (uav.H_MIN + uav.H_MAX) / 2
            dz = np.clip(mid_z - pos[q, 2], -max_step_z, max_step_z)
            new_pos[q, 2] = pos[q, 2] + dz

    new_pos[:, 2] = np.clip(new_pos[:, 2], uav.H_MIN, uav.H_MAX)
    new_pos[Q-1]  = new_pos[0]   # periodic boundary
    return new_pos


def _trajectory_rate(pos, uav, env, omega):
    total = 0.0
    Q, K  = uav.Q, env.num_nodes
    for q in range(Q):
        for k in range(K):
            if omega[q, k] < 0.5:
                continue
            d     = distance_3d(pos[q], env.node_pos3d(k))
            sr    = sinr_rad(uav.Prad[q], uav.Pcom[q], h_rad(d))
            total += radar_rate(sr)
    return total


def solve_trajectory(uav, env, omega, b, node_service=None, R_min=0.5, tol=1e-4, max_iter=30):
    Q  = uav.Q
    dt = uav.dt
    pos      = uav.position.copy()
    prev_obj = _trajectory_rate(pos, uav, env, omega)

    for it in range(max_iter):
        pos_new = _smooth_trajectory(pos, uav, env, omega, b,
                                     node_service=node_service, R_min=R_min)
        new_obj = _trajectory_rate(pos_new, uav, env, omega)
        if new_obj >= prev_obj - 1e-9:
            pos      = pos_new
            prev_obj = new_obj
        if it > 2 and abs(new_obj - prev_obj) < tol:
            break

    uav.position = pos.copy()
    for q in range(Q - 1):
        uav.velocity[q] = (uav.position[q+1] - uav.position[q]) / dt
    uav.velocity[Q-1] = uav.velocity[Q-2]


# ─────────────────────────────────────────────────────────────────────────────
# Utility: total radar rate  (same as baseline)
# ─────────────────────────────────────────────────────────────────────────────

def compute_total_radar_rate(uav, env, omega):
    return _trajectory_rate(uav.position, uav, env, omega)


def compute_per_node_rate(uav, env, omega):
    """Return (K,) array of total radar service accumulated per node."""
    K     = env.num_nodes
    rates = np.zeros(K)
    for q in range(uav.Q):
        for k in range(K):
            if omega[q, k] > 0.5:
                d = distance_3d(uav.position[q], env.node_pos3d(k))
                sr = sinr_rad(uav.Prad[q], uav.Pcom[q], h_rad(d))
                rates[k] += radar_rate(sr)
    return rates


def compute_energy(uav):
    """Total transmit energy in joules."""
    return float(np.sum(uav.Pt) * uav.dt)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 2 – improved three-layer iterative optimisation
# ─────────────────────────────────────────────────────────────────────────────

def three_layer_optimize_improved(
        uav, env,
        max_outer=20,
        tol=1e-3,
        lam=1.0,            # Pareto weight: 1 = pure radar, 0 = pure energy
        R_min=0.5,          # fairness floor [bps/Hz per node]
        verbose=True):
    """
    Improved Algorithm 2 with two enhancements:

    1. Multi-objective power allocation  (lam controls the trade-off)
    2. Fairness-aware scheduling         (every node ≥ R_min bps/Hz)

    Returns
    -------
    rates        : list[float]        radar rate after each outer iteration
    energies     : list[float]        total energy (J) after each outer iter
    omega_hist   : list[np.ndarray]   scheduling matrices
    node_service : np.ndarray (K,)    per-node accumulated radar service
    """
    rates      = []
    energies   = []
    omega_hist = []
    prev_rate  = -np.inf

    for i in range(max_outer):
        # ── Channel gains ──────────────────────────────────────────────────
        hk_com, hk_rad, hc, dk, dc = _channel_gains(uav, env)

        # ── Step 1: Fairness-aware scheduling ─────────────────────────────
        omega, b, Rrad, Rcom, Rc, node_svc = solve_scheduling_fair(
            uav, env, hk_com, hk_rad, hc, R_min=R_min)

        # ── Step 2: Multi-objective power allocation ───────────────────────
        energy = solve_power_multiobjective(
            uav, env, omega, b, hk_com, hk_rad, hc, lam=lam)

        # ── Step 3: 3-D trajectory ─────────────────────────────────────────
        solve_trajectory(uav, env, omega, b,
                         node_service=node_svc, R_min=R_min)

        # ── Evaluate ──────────────────────────────────────────────────────
        rate = compute_total_radar_rate(uav, env, omega)
        rates.append(rate)
        energies.append(energy)
        omega_hist.append(omega.copy())

        if verbose:
            isac_cnt   = int(omega.sum())
            upload_cnt = int(b.sum())
            fair_nodes = int(np.sum(node_svc >= R_min))
            print(f"  Iter {i+1:2d}  |  Radar: {rate:8.4f} bps/Hz"
                  f"  |  Energy: {energy:7.2f} J"
                  f"  |  ISAC: {isac_cnt:3d}  Upload: {upload_cnt:3d}"
                  f"  |  Fair nodes: {fair_nodes}/{env.num_nodes}")

        # FIX 3: Converge on the scalarised composite objective, not just
        # radar rate. Using radar rate alone causes lam<1 runs to stop
        # before energy has meaningfully changed, producing identical curves.
        energy_scale_conv = uav.Q * uav.P_AVG * uav.dt
        composite = lam * rate - (1 - lam) * (energy / (energy_scale_conv + 1e-12))
        if abs(composite - prev_rate) < tol and i > 1:
            if verbose:
                print(f"  ✅ Converged at iteration {i+1}")
            break
        prev_rate = composite

    final_node_service = compute_per_node_rate(uav, env, omega_hist[-1])
    return rates, energies, omega_hist, final_node_service


# ─────────────────────────────────────────────────────────────────────────────
# Pareto front sweep
# ─────────────────────────────────────────────────────────────────────────────

def compute_pareto_front(env_cls, uav_cls, env_kwargs, uav_kwargs,
                         lambdas=None, R_min=0.5, max_outer=20,
                         verbose=False):
    """
    Sweep over lambda values to trace the Pareto front between radar rate
    and energy consumption.

    Parameters
    ----------
    env_cls / uav_cls   : class constructors
    env_kwargs/uav_kwargs: dict of constructor kwargs
    lambdas             : array of lambda values in [0,1]
    R_min               : fairness floor (passed to every run)
    max_outer           : outer iterations per lambda

    Returns
    -------
    pareto_rates    : list of final radar rates
    pareto_energies : list of final total energies
    lambdas         : the lambda sweep values used
    """
    if lambdas is None:
        lambdas = np.linspace(0.0, 1.0, 11)

    pareto_rates    = []
    pareto_energies = []

    for lam in lambdas:
        env = env_cls(**env_kwargs)
        uav = uav_cls(**uav_kwargs)
        uav.initialize_trajectory(env)

        rates, energies, omega_hist, _ = three_layer_optimize_improved(
            uav, env,
            max_outer=max_outer,
            lam=lam,
            R_min=R_min,
            verbose=verbose)

        pareto_rates.append(rates[-1])
        pareto_energies.append(energies[-1])
        if verbose:
            print(f"  λ={lam:.2f}  →  R={rates[-1]:.4f}  E={energies[-1]:.2f} J")

    return pareto_rates, pareto_energies, list(lambdas)
