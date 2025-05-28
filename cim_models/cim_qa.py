"""
File: cim_qa.py
Author: Vidisha Singhal
Created: 2025-05-25
Description: This module provides the numerical implementation of the standard Coherent Ising Machine (CIM) with Quantum Adiabaticity (QA) model
"""

import numpy as np

def cim_qa(x0, alpha, p, Jp, noise_level, coupling_coeff, dt, T, N, M):
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
    - M: Number of steps for the interpolation between Jb and Jp
    """
    num_steps = int(T / dt)
    states = np.zeros((num_steps + 1, N))
    states[0] = x0
    x = x0
    
    Jb = np.ones((N, N)) - np.eye(N)            #defining initial Hamiltonian (all 1s with no self-connections)
    J_t = Jb                                    #setting initial hamiltonian

    print(Jp)

    interval = num_steps // M                   #computing when hamiltonian needs to be updated

    for step in range(num_steps):
        if step % interval == 0:
          m = step // interval
          lam = m / M
          if (m == M - 1):
            J_t = Jp
          else:
            J_t = (1 - lam) * Jb + lam * Jp     #interpolating between Jb and Jp
          print(J_t)
        
        I_inj = coupling_coeff * np.dot(J_t, x)
        dx_dt = (p - 1) * x - alpha * x**3 + I_inj
        noise = noise_level * np.sqrt(dt) * np.random.normal(-1, 1, N)
        x = x + (dx_dt * dt) + noise
        states[step + 1] = x
    
    return states, x
