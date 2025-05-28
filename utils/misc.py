import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import os

def create_partition_from_opo_amplitudes(x):
    """
    Create a partition array based on the sign of the OPO amplitudes
    Input: Amplitudes (numpy array of size N)
    Returns: Binary array where -1 and 1 represent the two sets in the partition
    """
    partition = np.where(x >= 0, 1, -1)    
    return partition

def evaluate_maxcut_value_from_partition(coupling_matrix, partition):
    """    
    Inputs:
    - coupling_matrix: A square matrix where element (i, j) represents the weight of the edge between nodes i and j.
    - partition: Array where -1 and 1 represent the two sets in the partition

    Returns:
    - The cut value of the partition, which is the sum of weights of edges between the two sets
    """
    N = coupling_matrix.shape[0]
    if coupling_matrix.shape[1] != N or len(partition) != N:
        raise ValueError("Coupling matrix dimensions and partition length must match.")
    
    set1_indices = np.where(partition == -1)[0]
    set2_indices = np.where(partition == 1)[0]
    
    cut_value = 0
    for i in set1_indices:
        for j in set2_indices:
            cut_value += coupling_matrix[i, j]

    print(f"Cut value: {cut_value}")

    return cut_value


def combine_cim_results_to_excel(cim_type):
    folder_path = os.path.join("results", "optimization_results", cim_type)
    excel_output_path = os.path.join("results", f"{cim_type}_results.xlsx")
    
    with pd.ExcelWriter(excel_output_path, engine="openpyxl") as writer:
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                json_path = os.path.join(folder_path, filename)
                with open(json_path, "r") as f:
                    data = json.load(f)
                
                df = pd.json_normalize(data)

                sheet_name = os.path.splitext(filename)[0]
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"Combined Excel file created at: {excel_output_path}")


def plot_states(states, dt):
    """
    Plot the variable states as a function of time.
    
    Parameters:
    - states: Array of system states over time
    - dt: Time step size for the simulation
    """
    num_steps = states.shape[0]
    time = np.arange(0, num_steps * dt, dt)
    
    plt.figure(figsize=(12, 8))
    
    # Plot each state's variable over time
    for i in range(states.shape[1]):
        plt.plot(time, states[:, i], label=f'Node {i}')
    
    plt.xlabel('Time')
    plt.ylabel('State Value')
    plt.title('States vs Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def brute_force_maxcut(coupling_matrix):
    """    
    Parameters:
    - coupling_matrix: An n x n matrix where element (i, j) represents the weight of the edge between nodes i and j.
    
    """
    N = coupling_matrix.shape[0]
    max_cut_value = -float('inf')
    best_partition = None
    
    # Generate all possible partitions
    num_partitions = 2 ** N
    for i in range(num_partitions):
        # Create a partition array where binary representation of i determines the partition
        partition = np.array([1 if (i >> j) & 1 else -1 for j in range(N)])
        current_cut_value = evaluate_maxcut_value(coupling_matrix, partition)
        if current_cut_value > max_cut_value:
            max_cut_value = current_cut_value
            best_partition = partition
    
    return max_cut_value