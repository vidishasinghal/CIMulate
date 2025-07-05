"""
File: cim_fon.py
Author: Vidisha Singhal
Created: 2025-05-25
Description: This module provides the numerical implementation of the standard Coherent Ising Machine (CIM) with Fifth-Order Non-Linearity (FON) model
"""

import numpy as np
import cupy as cp
import time

steps_save_interval = 100

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
    num_state_saves = (num_steps // steps_save_interval)
    states = np.zeros((num_state_saves, N))   #saving values of x at each time step here, used to plot evolution of OPOs vs time
    states[0] = x0
    x = x0
    save_idx = 0

    start_time = time.time()
    
    for step in range(num_steps):
        I_inj = coupling_coeff * np.dot(J, x)
        
        dx_dt = (p - 1) * x - (alpha * x**3) + (beta * x**5) + I_inj    #adding the fifth-order term in addition to standard cim
        noise = noise_level * np.sqrt(dt) * np.random.normal(-1, 1, N)
        
        x = x + (dx_dt * dt) + noise

        if step % steps_save_interval == 0:
            states[save_idx] = x
            save_idx += 1

    end_time = time.time()
    simulation_time = end_time - start_time

    return states, x, simulation_time



def cim_fon_gpu(x0, J, noise_level, dt, T, N, alpha, p, coupling_coeff, beta):
    """
    GPU-accelerated version of the CIM-FON model using CuPy
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
    #print("Running standard CIM on GPU (large N)...")

    # Move data to GPU
    x0_gpu = cp.array(x0)
    J_gpu = cp.array(J)

    num_steps = int(T / dt)
    num_state_saves = (num_steps // steps_save_interval) + 1
    states = cp.zeros((num_state_saves, N))       #saving values of x at each time step here, used to plot evolution of OPOs vs time
    states[0] = x0_gpu
    x = x0_gpu                                  #initializing x to the initial random state of OPOs
    save_idx = 0

    start_time = time.time()
    noise = noise_level * cp.sqrt(dt) * cp.random.normal(-1, 1, size=(num_steps, N))
        
    for step in range(num_steps):
        #print(f"Step {step}!")
        I_inj = coupling_coeff * cp.dot(J_gpu, x)

        #dx_dt = (p - 1) * x - alpha * x**3 + I_inj
        #noise = noise_level * cp.sqrt(dt) * cp.random.normal(-1, 1, N)

        #x = x + (dx_dt * dt) + noise[step]

        x = fused_update(x, I_inj, noise[step], dt, alpha, p, beta)

        if step % steps_save_interval == 0:
            states[save_idx] = x
            save_idx += 1

    # Move results back to CPU
    x = cp.asnumpy(x)
    states = cp.asnumpy(states)

    end_time = time.time()
    simulation_time = end_time - start_time

    return states, x, simulation_time


@cp.fuse()
def fused_update(x, I_inj, noise, dt, alpha, p, beta):
    return (x + (((p - 1) * x - (alpha * x**3) + (beta * x**5) + I_inj) * dt) + noise)