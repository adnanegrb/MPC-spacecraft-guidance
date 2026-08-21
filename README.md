# 🛸 MPC Spacecraft Guidance

![Language](https://img.shields.io/badge/Language-Python-blue)
![Topic](https://img.shields.io/badge/Topic-Optimal%20Control-purple)
![Domain](https://img.shields.io/badge/Domain-Aerospace-darkblue)
![Methods](https://img.shields.io/badge/Methods-Linear%20MPC%20%7C%20Tube%20MPC%20%7C%20Fast%20MPC%20%7C%20SCvx-orange)
![Dynamics](https://img.shields.io/badge/Dynamics-Clohessy--Wiltshire-green)

> Real-time Model Predictive Control for autonomous spacecraft guidance. Implementing Linear MPC, Tube MPC, Fast MPC and Successive Convexification from scratch, with no optimization library, and reproducing the reference examples from the underlying papers.

## Context

This repository provides clean Python implementations of real-time MPC algorithms for autonomous spacecraft guidance and orbital rendezvous. It covers four algorithmic families: Linear MPC, Tube MPC for formal robustness guarantees under bounded disturbances, Fast MPC for embedded-oriented solvers, and Successive Convexification for nonlinear problems. Every solver is implemented from scratch using only basic linear algebra, with no external optimization library, and validated against the examples presented in the corresponding papers.

## Structure

```
algos/        core solver for each algorithmic family
solvers/       low-level QP solvers shared across algorithms (ADMM, Nesterov)
examples/      one script per paper, reproducing its reference example
utils/         shared dynamics utilities
tests/         correctness and closed-loop guarantee tests (pytest)
configs/       example parameter set, to adapt to your own problem
```

## Installation

```bash
git clone https://github.com/adnanegrb/MPC-spacecraft-guidance.git
cd MPC-spacecraft-guidance
pip install -r requirements.txt
```

## Testing

```bash
pip install pytest
pytest tests/
```

Tests cover solver correctness (ADMM and Nesterov agree on the same QP,
both respect the thrust bound), and closed-loop guarantees (the LQR gain
stabilises the error dynamics, the tracking error stays bounded under
bounded disturbances, the Hessian conditioning grows with the horizon
as discussed in the report).

## 🔧 Configuration

`configs/example_params.yaml` is one example parameter set used in the
report, not a fixed template. Every field is commented with its role, so
you can swap in your own orbit, horizon, weights, or disturbance levels
for your own problem.

## Core Mathematics

**Dynamics.** The chaser's relative motion follows the linearised
Clohessy-Wiltshire-Hill equations, $\dot{\boldsymbol{x}} = A_c\boldsymbol{x} + B_c\boldsymbol{u}$,
discretised exactly via the matrix exponential:

$$\boldsymbol{x}_{k+1} = A_d \boldsymbol{x}_k + B_d \boldsymbol{u}_k + \boldsymbol{\xi}_k, \qquad A_d = e^{A_c T_s}, \quad B_d = \int_0^{T_s} e^{A_c \tau} B_c \, d\tau$$

**Condensed QP.** Every MPC family reduces, online, to the same
box-constrained quadratic program in the stacked control sequence
$\boldsymbol{v}$:

$$\min_{\boldsymbol{v}} \ \tfrac{1}{2}\boldsymbol{v}^\top H \boldsymbol{v} + \boldsymbol{f}^\top \boldsymbol{v} \quad \text{s.t.} \quad \|\boldsymbol{v}_i\|_\infty \le u_{\max}, \qquad H = \mathcal{B}^\top \bar{Q} \mathcal{B} + \bar{R}$$

**Tube MPC error dynamics.** Splitting the true state $\boldsymbol{x}_k = \boldsymbol{z}_k + \boldsymbol{e}_k$
into a nominal part and an LQR-corrected error decouples the dynamics:

$$\boldsymbol{e}_{k+1} = A_{cl}\boldsymbol{e}_k + \boldsymbol{\xi}_k, \qquad A_{cl} = A_d - B_d K, \qquad K = (R + B_d^\top P B_d)^{-1}B_d^\top P A_d$$

with $P$ the solution of the discrete algebraic Riccati equation. Since
$A_{cl}$ is stable, the error stays inside a bounded invariant set, and
the corridor constraint reduces to a tightened constraint on $\boldsymbol{z}_k$
alone, at no extra online cost.

**Conditioning.** Because CWH is marginally stable ($|\lambda_i(A_d)| \approx 1$), the condensed Hessian's condition number $\kappa(H)$ grows sharply with the horizon $N$, which is why first-order solvers (Nesterov) can stall where direct solves (ADMM via Cholesky) remain unaffected, the central empirical finding reproduced in `examples/`.



## Main Key References

**Hartley (2015)** : A Tutorial on Model Predictive Control for Spacecraft Rendezvous.

**Breger & How (2008)** : Safe Trajectories for Autonomous Rendezvous of Spacecraft.

**Di Cairano, Park & Kolmanovsky (2012)** : MPC Approach for Guidance of Spacecraft Rendezvous and Proximity Maneuvering.

**Specht, Bishnoi & Lampariello (2023)** : Autonomous Spacecraft Rendezvous Using Tube-Based MPC: Design and Application.

**Oestreich, Linares & Gondhalekar (2023)** : Tube-Based MPC with Uncertainty Identification for Autonomous Spacecraft Maneuvers.

**Richter, Jones & Morari (2012)** : Computational Complexity Certification for Real-Time MPC With Input Constraints Based on the Fast Gradient Method.

**Szmuk, Acikmese & Reynolds (2020)** : Successive Convexification for Passively-Safe Spacecraft Rendezvous on Near Rectilinear Halo Orbit.
