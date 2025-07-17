"""
File: cim_snn.py
Author: Vidisha Singhal
Created: 2025-05-25
Description: This module provides the numerical implementation of the Coherent Ising Machine (CIM) model with Spiking Neural Network (SNN)
Copyright (c) 2025 Vidisha Singhal
"""

import numpy as np
import cupy as cp
import time

steps_save_interval = 100

def cim_snn(x0, alpha, p, J, noise_level, coupling_coeff, dt, T, N, lambda_snn, linear_pump_schedule=None):
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
    num_state_saves = (num_steps // steps_save_interval)
    states = np.zeros((num_state_saves, N))
    states[0] = x0
    save_idx = 0
    
    x = x0

    b = np.zeros(N)                                     #dissipative pulse values

    start_time = time.time()

    for step in range(num_steps):
        if linear_pump_schedule is not None:
            p = linear_pump_schedule["start"] + (linear_pump_schedule["end"] - linear_pump_schedule["start"]) * (step / num_steps)

        I_inj = coupling_coeff * np.dot(J, x)
        
        dx_dt = (p - 1) * x - (alpha * x**3) - (lambda_snn * b) + I_inj
        db_dt = -lambda_snn * b + x
        noise = noise_level * np.sqrt(dt) * np.random.normal(-1, 1, N)
        
        x = x + (dx_dt * dt) + noise
        b = b + (db_dt * dt)                            #updating the dissipative pulse values as in original paper

        if step % steps_save_interval == 0:
            states[save_idx] = x
            save_idx += 1

    end_time = time.time()
    simulation_time = end_time - start_time

    return states, x, simulation_time



def cim_snn_gpu(x0, alpha, p, J, noise_level, coupling_coeff, dt, T, N, lambda_snn, linear_pump_schedule=None):
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
    #print("Running standard CIM on GPU (large N)...")

    # Move data to GPU
    x0_gpu = cp.array(x0)
    J_gpu = cp.array(J)

    num_steps = int(T / dt)
    num_state_saves = (num_steps // steps_save_interval)
    states = cp.zeros((num_state_saves, N))
    states[0] = x0_gpu
    save_idx = 0

    x = x0_gpu
    b = cp.zeros(N)                                     #dissipative pulse values

    start_time = time.time()

    noise = noise_level * cp.sqrt(dt) * cp.random.normal(-1, 1, size=(num_steps, N))

    if linear_pump_schedule is not None:
        pump_array = linear_pump_schedule["start"] + (linear_pump_schedule["end"] - linear_pump_schedule["start"]) * (cp.arange(num_steps) / (num_steps - 1))

    for step in range(num_steps):
        p = pump_array[step] if linear_pump_schedule is not None else p

        I_inj = coupling_coeff * cp.dot(J_gpu, x)

        x, b = fused_update(x, I_inj, noise[step], dt, alpha, p, b, lambda_snn)

        if step % steps_save_interval == 0:
            states[save_idx] = x
            save_idx += 1

    x = cp.asnumpy(x)
    states = cp.asnumpy(states)

    end_time = time.time()
    simulation_time = end_time - start_time

    return states, x, simulation_time


@cp.fuse()
def fused_update(x, I_inj, noise, dt, alpha, p, b, lambda_snn):
    x_update = (x + ((p - 1) * x - (alpha * x**3) - (lambda_snn * b) + I_inj) * dt + noise)
    b_update = b + (-lambda_snn * b + x) * dt
    return x_update, b_update