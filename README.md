# UAV-ISAC-Optimization

> 3-D Trajectory Optimisation and Multi-Objective Resource Allocation for UAV-Assisted Integrated Sensing and Communications

## Overview

This repository implements an improved UAV-ISAC optimisation framework for IoT deployments in
infrastructure-scarce environments. Built on the three-layer iterative algorithm of Liu et al.
(IEEE Trans. Wireless Commun., 2024), it introduces fairness-constrained scheduling,
multi-objective Pareto power allocation, and service-aware 3-D trajectory planning to eliminate
node starvation and reduce energy waste.

---

## Problem Statement

Existing UAV-ISAC frameworks maximise sum radar estimation rate — a greedy objective that starves
geometrically disadvantaged IoT nodes and ignores transmit energy consumption. In a 12-node
scenario, the baseline leaves 4 nodes with zero radar service and expends full transmit power
even when unnecessary.

---

## 🧾 Project Structure


---

## 🧠 Features

- Per-node fairness-constrained scheduling (R_min guarantee)
- Multi-objective Pareto power allocation with tunable weight λ
- Service-aware 3-D trajectory with deficit-based UAV attraction
- Convergence-guaranteed three-layer iterative optimisation
- Visualisations of trajectory, power, scheduling, and altitude profiles
- Comparative study between baseline (Liu et al.) and improved versions

---

## ⚙️ Algorithm Overview

The three-layer block-coordinate descent structure combines:

- **Fairness-Constrained Scheduling (Layer 1):** LP with per-node minimum-rate constraint ensures
  every IoT node receives ≥ R_min = 0.5 bps/Hz of radar service per flight cycle.
- **Multi-Objective Pareto Power Allocation (Layer 2):** SCA-based optimiser jointly maximises
  radar rate and minimises transmit energy via scalarised composite objective.
- **Service-Aware 3-D Trajectory (Layer 3):** Node-service tally redirects UAV towards
  under-served nodes using deficit-proportional attraction bonus.

| Layer | Improvement | Effect |
|-------|-------------|--------|
| Scheduling (L1) | Per-node minimum-rate fairness constraint | Eliminates node starvation |
| Power Allocation (L2) | Scalarised multi-objective Pareto formulation | Reduces transmit energy |
| Trajectory (L3) | Service-aware node-tally with deficit-based UAV attraction | Guides UAV to under-served nodes |

---

## 📈 Performance Comparison

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
| Sum Radar Rate | 47.38 bps/Hz | 33.68 bps/Hz | −29% |
| Transmit Energy | 100 J | 91 J | −9% |
| Nodes Fairly Served | 8 / 12 | **12 / 12** | +4 nodes |
| Node Starvation Rate | 33% | **0%** | −33 pp |
| Convergence Iterations | ~9 | ~15 | +6 iters |

Plots and logs are available in the `plots/` directory for detailed insights.

---

## 📦 Requirements

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

Typical dependencies include:

- numpy
- scipy
- matplotlib
- cvxpy
- pandas

---

## 🚀 How to Use

**1. Clone this repository:**

```bash
git clone https://github.com/Ragamaie-Nagineni/uav-isac-iot-optimiser.git
cd uav-isac-iot-optimiser
```

**2. Install required libraries:**

```bash
pip install -r requirements.txt
```

**3. Run the simulation:**

```bash
python simulations/run_experiment.py --lambda 0.8 --rmin 0.5 --nodes 12 --slots 200
```

**4. Tune parameters:**

```bash
# Recommended balanced setting
python simulations/run_experiment.py --lambda 0.8 --rmin 0.5

# Aggressive energy saving
python simulations/run_experiment.py --lambda 0.7 --rmin 0.5

# Recover exact baseline behaviour
python simulations/run_experiment.py --lambda 1.0 --rmin 0.0
```

**5. Analyse outputs:**

- Compare convergence curves (baseline vs improved)
- View trajectory and altitude profiles
- Evaluate per-node service fairness
- Inspect power allocation across time slots

---

## System Model & Parameters

- **Area:** 1200 × 1200 m²  
- **Nodes:** 12 IoT nodes, 1 Data Collection Centre (DCC)  
- **Flight:** T = 100 s, Q = 200 time slots  
- **UAV Payload:** Dual-Function Radar-Communication (DFRC)  
- **Channel:** Dominant LoS, path-loss exponents 2 (comm) and 4 (radar)

```python
K       = 12          # IoT nodes
Q       = 200         # Time slots
T       = 100         # Flight time (s)
Pavg    = 1           # Avg power budget (W)
Rmin    = 0.5         # Fairness floor (bps/Hz)
lambda  = 0.8         # Pareto weight (rate vs energy)
theta   = 30          # Radar detection half-angle (deg)
f_c     = 3.5e9       # Carrier frequency (Hz)
B       = 50e6        # Bandwidth (Hz)
```

---

## Trade-off Guide

| λ | Sum Rate (bps/Hz) | Energy (J) | Nodes Served |
|---|---|---|---|
| 1.0 (baseline) | 47.38 | 100.0 | 8/12 |
| 0.9 | 38.5 | 94.0 | 12/12 |
| **0.8 (recommended)** | **33.7** | **91.0** | **12/12** |
| 0.7 | 28.1 | 87.5 | 12/12 |
| 0.5 | 18.4 | 82.0 | 12/12 |

> Recommended range: **λ ∈ [0.75, 0.85]** for most emergency IoT deployments.

---

## Key Equations

**Radar Estimation Rate:**
$$R_{k,rad}(q) = \log_2(1 + \Gamma_{k,rad}(q))$$

**Multi-Objective Power Objective:**
$$\max \; \lambda \cdot R_{sum} - (1 - \lambda) \cdot \frac{E}{E_{scale}}$$

**Fairness Constraint:**
$$\sum_{q=1}^{Q} \omega_k(q) \, R_{k,rad}(q) \geq R_{min}, \quad \forall k$$

**Service-Aware Trajectory Bonus:**
$$\tilde{\xi}^{(i)}_{k,rad} = \xi^{(i)}_{k,rad} \cdot \left(1 + \gamma \max(R_{min} - s^{(i)}_k, 0)\right)$$

---

## 📌 Notes

- Based on academic research (Liu et al., IEEE Trans. Wireless Commun., 2024)
- For research and educational purposes only
- Improvements are experimental and can be extended further
- Static IoT node locations assumed known a priori
- Perfect LoS channel assumed (no shadowing or multipath)

---

## Future Work

- Mobile IoT nodes with Kalman-filter position prediction
- Multi-UAV coordination with distributed fairness constraints
- Deep reinforcement learning sub-solvers for non-stationary channels
- Joint waveform design for physical-layer security against eavesdroppers

---

## 📚 Reference

This project builds upon:

> Z. Liu, X. Liu, Y. Liu, V. C. M. Leung, and T. S. Durrani,
> "UAV Assisted Integrated Sensing and Communications for Internet of Things:
> 3D Trajectory Optimization and Resource Allocation,"
> *IEEE Trans. Wireless Commun.*, vol. 23, no. 8, pp. 8654–8667, Aug. 2024.

```bibtex
@article{liu2024uavisac,
  author  = {Z. Liu and X. Liu and Y. Liu and V. C. M. Leung and T. S. Durrani},
  title   = {UAV Assisted Integrated Sensing and Communications for Internet of Things},
  journal = {IEEE Trans. Wireless Commun.},
  volume  = {23},
  number  = {8},
  pages   = {8654--8667},
  year    = {2024}
}
```

---

## 👥 Contributors

This project was collaboratively developed by:

- [**Ragamaie Nagineni**](https://github.com/Ragamaie-Nagineni)
- [**G Sai Vyshnavi**](https://github.com/saivyshnavi)
- [**Boppana Goutam**](https://github.com/goutam-b)

Department of Computer Science and Engineering / Artificial Intelligence and Data Science
Indian Institute of Information Technology Sri City, Chittoor, Andhra Pradesh, India

---

## 📁 License

This project is open-source and distributed under the [MIT License](LICENSE).

---

## 🤝 Contribution

Contributions, issues, and feature requests are welcome!
Feel free to check the [issues page](https://github.com/Ragamaie-Nagineni/uav-isac-iot-optimiser/issues)
or submit a pull request.

---

## 📬 Contact

For questions or collaboration, reach out to:

- **Name:** Ragamaie Nagineni
- **Email:** [ragamaie.n@gmail.com](mailto:ragamaie.n@gmail.com)
- **GitHub:** [Ragamaie-Nagineni](https://github.com/Ragamaie-Nagineni)
