"""
File: cim_qa.py
Author: Vidisha Singhal
Created: 2025-05-25
Description: This module provides the numerical implementation of the standard Coherent Ising Machine (CIM) with Quantum Adiabaticity (QA) model
"""

import numpy as np
import cupy as cp
import time

steps_save_interval = 100

def cim_qa(x0, alpha, p, J, noise_level, coupling_coeff, dt, T, N, M, linear_pump_schedule=None):
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
    M = int(M)                                  #ensure M is an integer
    Jp = J
    num_steps = int(T / dt)
    num_state_saves = (num_steps // steps_save_interval)
    states = np.zeros((num_state_saves, N))
    states[0] = x0
    x = x0
    save_idx = 0
    
    Jb = np.ones((N, N)) - np.eye(N)            #defining initial Hamiltonian (all 1s with no self-connections)
    J_t = Jb                                    #setting initial hamiltonian

    interval = num_steps // M                   #computing when hamiltonian needs to be updated

    start_time = time.time()

    for step in range(num_steps):
        if step % interval == 0:
          m = step // interval
          lam = m / M
          if (m == M - 1):
            J_t = Jp
          else:
            J_t = (1 - lam) * Jb + lam * Jp     #interpolating between Jb and Jp
        
        if linear_pump_schedule is not None:
            pump_rate = linear_pump_schedule["start"] + (linear_pump_schedule["end"] - linear_pump_schedule["start"]) * (step / num_steps)

        I_inj = coupling_coeff * np.dot(J_t, x)
        dx_dt = (p - 1) * x - alpha * x**3 + I_inj
        noise = noise_level * np.sqrt(dt) * np.random.normal(-1, 1, N)
        x = x + (dx_dt * dt) + noise

        if step % steps_save_interval == 0:
            states[save_idx] = x
            save_idx += 1

    end_time = time.time()
    simulation_time = end_time - start_time

    return states, x, simulation_time



def cim_qa_gpu(x0, alpha, p, J, noise_level, coupling_coeff, dt, T, N, M, linear_pump_schedule=None):
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
    M = int(M)                                  #ensure M is an integer
    # Transfer data to GPU
    x = cp.array(x0)
    Jp = cp.array(J)
    Jb = cp.ones((N, N)) - cp.eye(N)

    num_steps = int(T / dt)
    num_state_saves = (num_steps // steps_save_interval)
    states = cp.zeros((num_state_saves, N))
    states[0] = x
    save_idx = 0
    #x = x0
    
    J_t = Jb                                    #setting initial hamiltonian

    interval = num_steps // M                   #computing when hamiltonian needs to be updated

    start_time = time.time()                    #starting before noise generation to be consistent with CPU version

    noise = noise_level * cp.sqrt(dt) * cp.random.normal(-1, 1, size=(num_steps, N))
    
    if linear_pump_schedule is not None:
      pump_array = linear_pump_schedule["start"] + (linear_pump_schedule["end"] - linear_pump_schedule["start"]) * (cp.arange(num_steps) / (num_steps - 1))


    for step in range(num_steps):
        if step % interval == 0:
          m = step // interval
          lam = m / M
          if (m == M - 1):
            J_t = Jp
          else:
            J_t = (1 - lam) * Jb + lam * Jp     #interpolating between Jb and Jp

        p = pump_array[step] if linear_pump_schedule is not None else p
                  
        I_inj = coupling_coeff * cp.dot(J_t, x)
        x = fused_update(x, I_inj, noise[step], dt, alpha, p)

        #dx_dt = (p - 1) * x - alpha * x**3 + I_inj
        #noise = noise_level * np.sqrt(dt) * np.random.normal(-1, 1, N)
        #x = x + (dx_dt * dt) + noise
        if step % steps_save_interval == 0:
            states[save_idx] = x
            save_idx += 1

    x = cp.asnumpy(x)
    states = cp.asnumpy(states)

    end_time = time.time()
    simulation_time = end_time - start_time

    return states, x, simulation_time


@cp.fuse()
def fused_update(x, I_inj, noise, dt, alpha, p):
    x = (x + ((p - 1) * x - alpha * x**3 + I_inj) * dt + noise)
    return x