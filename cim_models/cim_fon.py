"""
File: cim_fon.py
Author: Vidisha Singhal
Created: 2025-05-25
Description: This module provides the numerical implementation of the standard Coherent Ising Machine (CIM) with Fifth-Order Non-Linearity (FON) model
"""

import numpy as np

def cim_fon(x0, alpha, p, J, noise_level, coupling_coeff, dt, T, N, beta):
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
    - beta: coefficient for fifth-order term (x^5 term), unique to cim_fon
    """
    num_steps = int(T / dt)
    states = np.zeros((num_steps + 1, N))   #saving values of x at each time step here, used to plot evolution of OPOs vs time
    states[0] = x0
    x = x0
    
    for step in range(num_steps):
        I_inj = coupling_coeff * np.dot(J, x)
        
        dx_dt = (p - 1) * x - (alpha * x**3) + (beta * x**5) + I_inj    #adding the fifth-order term in addition to standard cim
        noise = noise_level * np.sqrt(dt) * np.random.normal(-1, 1, N)
        
        x = x + (dx_dt * dt) + noise
        
        states[step + 1] = x
    return states, x