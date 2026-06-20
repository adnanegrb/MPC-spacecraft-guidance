# 🛸 MPC Spacecraft Guidance

![Language](https://img.shields.io/badge/Language-Python-blue)
![Topic](https://img.shields.io/badge/Topic-Optimal%20Control-purple)
![Domain](https://img.shields.io/badge/Domain-Aerospace-darkblue)
![Methods](https://img.shields.io/badge/Methods-Linear%20MPC%20%7C%20Tube%20MPC%20%7C%20Fast%20MPC%20%7C%20SCvx-orange)
![Dynamics](https://img.shields.io/badge/Dynamics-Clohessy--Wiltshire-green)

> Real-time Model Predictive Control for autonomous spacecraft guidance. Implementing Linear MPC, Tube MPC, Fast MPC and Successive Convexification from scratch, with no optimization library, and reproducing the reference examples from the underlying papers.

## 📌 Context

This repository provides clean Python implementations of real-time MPC algorithms for autonomous spacecraft guidance and orbital rendezvous. It covers four algorithmic families: Linear MPC, Tube MPC for formal robustness guarantees under bounded disturbances, Fast MPC for embedded-oriented solvers, and Successive Convexification for nonlinear problems. Every solver is implemented from scratch using only basic linear algebra, with no external optimization library, and validated against the examples presented in the corresponding papers.

## 🗂 Structure

```
algos/        core solver for each algorithmic family
examples/     one script per paper, reproducing its reference example
utils/        shared dynamics utilities
```

## ⚙️ Installation

```bash
git clone https://github.com/adnanegrb/MPC-spacecraft-guidance.git
cd MPC-spacecraft-guidance
pip install -r requirements.txt
```

## 📚 Main Key References

**Hartley (2015)** : A Tutorial on Model Predictive Control for Spacecraft Rendezvous.

**Breger & How (2008)** : Safe Trajectories for Autonomous Rendezvous of Spacecraft.

**Di Cairano, Park & Kolmanovsky (2012)** : MPC Approach for Guidance of Spacecraft Rendezvous and Proximity Maneuvering.

**Specht, Bishnoi & Lampariello (2023)** : Autonomous Spacecraft Rendezvous Using Tube-Based MPC: Design and Application.

**Oestreich, Linares & Gondhalekar (2023)** : Tube-Based MPC with Uncertainty Identification for Autonomous Spacecraft Maneuvers.

**Richter, Jones & Morari (2012)** : Computational Complexity Certification for Real-Time MPC With Input Constraints Based on the Fast Gradient Method.

**Szmuk, Acikmese & Reynolds (2020)** : Successive Convexification for Passively-Safe Spacecraft Rendezvous on Near Rectilinear Halo Orbit.
