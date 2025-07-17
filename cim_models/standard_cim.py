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

steps_save_interval = 100

def standard_cim(x0, J, noise_level, dt, T, N, alpha, p, coupling_coeff, linear_pump_schedule=None):
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
    num_state_saves = (num_steps // steps_save_interval)
    states = None
    states = np.zeros((num_state_saves, N))   #saving values of x at each time step here, used to plot evolution of OPOs vs time
    states[0] = x0
    x = x0                                  #initializing x to the initial random state of OPOs
    save_idx = 0
        
    start_time = time.time()
    
    for step in range(num_steps):
        if linear_pump_schedule is not None:
            p = linear_pump_schedule["start"] + (linear_pump_schedule["end"] - linear_pump_schedule["start"]) * (step / num_steps)

        I_inj = coupling_coeff * np.dot(J, x)

        dx_dt = (p - 1) * x - alpha * x**3 + I_inj
        noise = noise_level * np.sqrt(dt) * np.random.normal(-1, 1, N)

        x = x + (dx_dt * dt) + noise

        if step % steps_save_interval == 0:
            states[save_idx] = x
            save_idx += 1

    end_time = time.time()
    simulation_time = end_time - start_time

    return states, x, simulation_time


def standard_cim_gpu(x0, J, noise_level, dt, T, N, alpha, p, coupling_coeff, linear_pump_schedule=None):
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
    # Move data to GPU
    x0_gpu = cp.array(x0)
    J_gpu = cp.array(J)

    num_steps = int(T / dt)
    num_state_saves = (num_steps // steps_save_interval)
    states = cp.zeros((num_state_saves, N))       #saving values of x at each time step here, used to plot evolution of OPOs vs time
    states[0] = x0_gpu
    x = x0_gpu                                  #initializing x to the initial random state of OPOs
    save_idx = 0

    start_time = time.time()

    noise = noise_level * cp.sqrt(dt) * cp.random.normal(-1, 1, size=(num_steps, N))

    if linear_pump_schedule is not None:
        pump_array = linear_pump_schedule["start"] + (linear_pump_schedule["end"] - linear_pump_schedule["start"]) * (cp.arange(num_steps) / (num_steps - 1))

    for step in range(num_steps):

        p = pump_array[step] if linear_pump_schedule is not None else p
        
        I_inj = coupling_coeff * cp.dot(J_gpu, x)

        x = fused_update(x, I_inj, noise[step], dt, alpha, p)

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
def fused_update(x, I_inj, noise, dt, alpha, p):
    return (x + ((p - 1) * x - alpha * x**3 + I_inj) * dt + noise)