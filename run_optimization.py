"""
File: run_optimization.py
Author: Vidisha Singhal
Created: 2025-05-25
Description: This script runs bayesian optimization process for all Coherent Ising Machine (CIM) configurations and problem instances.
"""

import os
import json
from utils.misc import combine_cim_results_to_excel, print_device_info
from bayesian_optimization.optimizer import CIMOptimizer

cim_architectures = ["cfc", "cac", "standard", "snn", "fon", "qa"]

def main():
    print_device_info()

    init_points = 10            # Number of initial random points to sample before Bayesian optimization starts
    n_iter = 10                # Number of iterations for Bayesian optimization after initial points
    trials = 10                # Number of bayesian optimization trials to run for each configuration (E.g. x trials for each CIM variant and problem instance combination)

    for name in cim_architectures:
        config_path = f"config/config_optimizations/config_{name}.json"
        CIM = CIMOptimizer(config_path)
        CIMOptimizer.optimize_all(CIM, trials=trials)
    
    print("All scheduled bayesian optimizations completed successfully.")
    
if __name__ == "__main__":
    main()