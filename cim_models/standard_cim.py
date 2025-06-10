"""
File: standard_cim.py
Author: Vidisha Singhal
Created: 2025-05-25
Description: This module provides the numerical implementation of the standard Coherent Ising Machine (CIM) model as described in the 2014 paper by Yamamoto et al.
Copyright (c) 2025 Vidisha Singhal
"""

import numpy as np
import cupy as cp
import time
from utils import misc

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

    #print("Running standard CIM on CPU (small N)...")

    num_steps = int(T / dt)
    states = None
    #states = np.zeros((num_steps + 1, N))   #saving values of x at each time step here, used to plot evolution of OPOs vs time
    #states[0] = x0
    x = x0                                  #initializing x to the initial random state of OPOs
        
    start_time = time.time()
    
    # Print whether running on GPU or CPU

    for step in range(num_steps):
        #print(f"Step {step}!")
        I_inj = coupling_coeff * np.dot(J, x)

        dx_dt = (p - 1) * x - alpha * x**3 + I_inj
        noise = noise_level * np.sqrt(dt) * np.random.normal(-1, 1, N)

        x = x + (dx_dt * dt) + noise

        #states[step + 1] = x
    
    end_time = time.time()
    print(f"Total time taken for {num_steps} steps: {end_time - start_time} seconds")

    return states, x


def standard_cim_gpu(x0, J, noise_level, dt, T, N, alpha, p, coupling_coeff):
    """
    GPU-accelerated version of the standard CIM model using CuPy
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
    #states = cp.zeros((num_steps + 1, N))       #saving values of x at each time step here, used to plot evolution of OPOs vs time
    #states[0] = x0_gpu
    x = x0_gpu                                  #initializing x to the initial random state of OPOs

    noise = noise_level * cp.sqrt(dt) * cp.random.normal(-1, 1, size=(num_steps, N))
    
    start_time = time.time()
    
    for step in range(num_steps):
        #print(f"Step {step}!")
        I_inj = coupling_coeff * cp.dot(J_gpu, x)

        #dx_dt = (p - 1) * x - alpha * x**3 + I_inj
        #noise = noise_level * cp.sqrt(dt) * cp.random.normal(-1, 1, N)

        #x = x + (dx_dt * dt) + noise[step]

        x = fused_update(x, I_inj, noise[step], dt, alpha, p)

        #states[step + 1] = x

    end_time = time.time()
    print(f"Total time taken for {num_steps} steps: {end_time - start_time} seconds")
    
    # Move results back to CPU
    #states = cp.asnumpy(states)
    x = cp.asnumpy(x)
    states = None

    return states, x


@cp.fuse()
def fused_update(x, I_inj, noise, dt, alpha, p):
    return (x + ((p - 1) * x - alpha * x**3 + I_inj) * dt + noise)