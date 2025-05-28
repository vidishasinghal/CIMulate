"""
File: standard_cim.py
Author: Vidisha Singhal
Created: 2025-05-25
Description: This module provides the numerical implementation of the standard Coherent Ising Machine (CIM) model as described in the 2014 paper by Yamamoto et al.
Copyright (c) 2025 Vidisha Singhal
"""

import numpy as np

def standard_cim(x0, J, noise_level, dt, T, N, alpha, p, coupling_coeff):
    """
    params:
    - x0: initial state of the system (numpy array of size N, random values)
    - J: coupling matrix of the graph (numpy array of size (N, N))
    - noise_level: coefficient for the noise term (eta in the paper)
    - dt: time step size for Euler-Maruyama integration
    - T: total integration time
    - N: number of nodes in graph / network
    - alpha: coefficient for the cubic nonlinearity (x^3 term)
    - p: constant pump rate paramter (can test linear or other functions as extension to this work)
    - coupling_coeff: strenght of coupling (xi in the paper)
    """
    num_steps = int(T / dt)
    states = np.zeros((num_steps + 1, N))   #saving values of x at each time step here, used to plot evolution of OPOs vs time
    states[0] = x0
    x = x0                                  #initializing x to the initial random state of OPOs
    
    for step in range(num_steps):
        I_inj = coupling_coeff * np.dot(J, x)

        dx_dt = (p - 1) * x - alpha * x**3 + I_inj
        noise = noise_level * np.sqrt(dt) * np.random.normal(-1, 1, N)

        x = x + (dx_dt * dt) + noise

        states[step + 1] = x

    return states, x