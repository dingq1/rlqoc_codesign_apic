# mmqoc_codesign_apic  


**Code and datasets for the research paper:**  

This repository contains the code used in the preprint paper below:

> **Hardware Co-Designed Optimal Control for Programmable Atomic Quantum Processors via Reinforcement Learning**  
> *[Qian Ding, Dirk Englund]*  
> *[arXiv, 2025 (link to be added !)]*  


##### Project Description
This project focuses on implementing hardware co-designed quantum optimal control (QOC) using Reinforcement Learning (RL) techniques on neutral atom platforms using programmable Photonic Integrated Circuits (PICs). The goal is to demonstrate robust, high-fidelity gate operations considering practical hardware with control imperfections like inter-channel crosstalk and beam leakage.

##### QOC Optimization Methods 
The code is written in JAX and we implement three quantum control optimization algorithms: 
1) Classical hybrid optimizer combining Self-Adaptive Differential Evolution (SaDE) and Adam (SADE-Adam)
2) Conventional Proximal policy optimization (PPO) based RL approach
3) End-to-End differentiable RL-based approach

##### Repository Structure
1) notebooks/          # Jupyter Notebooks for running experiments
2) results/            # code for plotting the results in figures
3) src/                # Source code for local and global control  
