"""
File: cim_snn.py
Author: Vidisha Singhal
Created: 2025-05-25
Description: This module provides the numerical implementation of the Coherent Ising Machine (CIM) model with Spiking Neural Network (SNN)
Copyright (c) 2025 Vidisha Singhal
"""

import numpy as np

def cim_snn(x0, alpha, p, J, noise_level, coupling_coeff, dt, T, N, lambda_snn):
    """
    params:
    - x0: initial state of the system (numpy array of size N, random values)
    - alpha: coefficient for the cubic nonlinearity (x^3 term)
    - p: constant pump rate paramter (can test linear or other functions as extension to this work)
    - J: coupling matrix of the graph (numpy array of size (N, N))
    - noise_level: coefficient for the noise term (eta in the paper)
    - coupling_coeff: strenght of coupling (xi in the paper)
    - dt: time step size for Euler-Maruyama integration
    - T: total integration time
    - lambda: spiking NN parameter (unique to cim_snn)
    """
    num_steps = int(T / dt)
    states = np.zeros((num_steps + 1, N))
    states[0] = x0
    
    x = x0

    b = np.zeros(N)                                     #dissipative pulse values

    for step in range(num_steps):
        I_inj = coupling_coeff * np.dot(J, x)
        
        dx_dt = (p - 1) * x - (alpha * x**3) - (lambda_snn * b) + I_inj
        db_dt = -lambda_snn * b + x
        noise = noise_level * np.sqrt(dt) * np.random.normal(-1, 1, N)
        
        x = x + (dx_dt * dt) + noise
        b = b + (db_dt * dt)                            #updating the dissipative pulse values as in original paper
        
        states[step + 1] = x
    
    return states, x