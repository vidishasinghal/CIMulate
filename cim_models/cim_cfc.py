"""
File: cim_cfc.py
Author: Vidisha Singhal
Created: 2025-05-25
Description: This module provides the numerical implementation of the standard Coherent Ising Machine (CIM) model with Chaotic Feedback Control
Copyright (c) 2025 Vidisha Singhal
"""

import numpy as np
import cupy as cp
import time

steps_save_interval = 100

def cim_cfc(x0, alpha, p, J, noise_level, coupling_coeff, dt, T, N, c_cfc, rho_cfc):
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
    - rho_cfc: parameter for the CFC model that is optimized in this work
    - c_cfc: another parameter for the CFC model that can be optimized (set to 1 for all tests in this work for consistency)
    """
       
    num_steps = int(T / dt)
    num_state_saves = (num_steps // steps_save_interval)
    states = np.zeros((num_state_saves, N))
    save_idx = 0

    x = x0
    e = -np.ones(N)
    states[0] = x0

    start_time = time.time()

    for step in range(num_steps):
        I_inj = -e * coupling_coeff * np.dot(J, x)
        
        dx_dt = (p - 1) * x - (alpha * x**3) + I_inj

        de_dt = -rho_cfc * e * (I_inj**2 - c_cfc)           #tracking the injected current instead of the x^2 term as in the CAC model
        
        noise = noise_level * np.sqrt(dt) * np.random.normal(-1, 1, N)
        
        x = x + (dx_dt * dt) + noise
        e = e + (de_dt * dt)

        if step % steps_save_interval == 0:
            states[save_idx] = x
            save_idx += 1

    end_time = time.time()
    simulation_time = end_time - start_time

    return states, x, simulation_time



def cim_cfc_gpu(x0, alpha, p, J, noise_level, coupling_coeff, dt, T, N, c_cfc, rho_cfc):
    
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
    - rho_cfc: parameter for the CAC model that is optimized in this work
    - c_cfc: another parameter for the CAC model that can be optimized (set to 1 for all tests in this work for consistency)
    """

    # Move data to GPU
    x0_gpu = cp.array(x0)
    J_gpu = cp.array(J)

    num_steps = int(T / dt)
    num_state_saves = (num_steps // steps_save_interval)
    save_idx = 0
    states = cp.zeros((num_state_saves, N))
    #states_e = np.zeros((num_steps + 1, N))
    states[0] = x0_gpu
    #states_e[0] = np.random.uniform(-0.001, 0.001, N)

    e = -cp.ones(N)
    x = x0_gpu

    #e = states_e[0]

    start_time = time.time()
    noise = noise_level * cp.sqrt(dt) * cp.random.normal(-1, 1, size=(num_steps, N))

    for step in range(num_steps):
        I_inj = -e * coupling_coeff * cp.dot(J_gpu, x)

        x, e = fused_update(x, I_inj, noise[step], dt, alpha, p, e, rho_cfc, c_cfc)
        
        if step % steps_save_interval == 0:
            states[save_idx] = x
            save_idx += 1
        #states_e[step + 1] = e
    
    x = cp.asnumpy(x)
    states = cp.asnumpy(states)

    end_time = time.time()
    simulation_time = end_time - start_time

    return states, x, simulation_time


@cp.fuse()
def fused_update(x, I_inj, noise, dt, alpha, p, e, rho_cfc, c_cfc):
    x_update = (x + (((p - 1) * x - (alpha * x**3) + I_inj) * dt) + noise)
    e_update = e + ((-rho_cfc * e * (I_inj**2 - c_cfc)) * dt)
    return x_update, e_update