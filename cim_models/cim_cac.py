"""
File: cim_cac.py
Author: Vidisha Singhal
Created: 2025-05-25
Description: This module provides the numerical implementation of the standard Coherent Ising Machine (CIM) model with Chaotic Amplitude Control (CAC)
Copyright (c) 2025 Vidisha Singhal
"""

import numpy as np

def cim_cac(x0, alpha, p, J, noise_level, coupling_coeff, dt, T, N, c_cac, rho_cac):
    
    """
    parameters:
    - x0: Initial state of the system (numpy array of size N)
    - alpha: Coefficient for the cubic nonlinearity
    - p: Parameter affecting the linear term
    - J: Coupling matrix (numpy array of size (N, N))
    - eta: Standard deviation of the noise
    - xi: Parameter for the injected current
    - dt: Time step size for Euler-Maruyama integration
    - T: Total integration time
    - N: Number of nodes in the network
    - rho_cac: parameter for the CAC model that is optimized in this work
    - c_cac: another parameter for the CAC model that can be optimized (set to 1 for all tests in this work for consistency)
    """

    num_steps = int(T / dt)
    states = np.zeros((num_steps + 1, N))
    states_e = np.zeros((num_steps + 1, N))
    states[0] = x0
    states_e[0] = np.random.uniform(-0.001, 0.001, N)
    
    x = x0
    e = states_e[0]

    for step in range(num_steps):
        I_inj = -e * coupling_coeff * np.dot(J, x)
        
        dx_dt = (p - 1) * x - (alpha * x**3) + I_inj

        de_dt = -rho_cac * e * (x**2 - c_cac)
        
        noise = noise_level * np.sqrt(dt) * np.random.normal(-1, 1, N)
        
        x = x + (dx_dt * dt) + noise
        e = e + (de_dt * dt)
        
        states[step + 1] = x
        states_e[step + 1] = e
    
    return states, x