"""
File: run_experiments.py
Author: Vidisha Singhal
Created: 2025-06-29
Description: This script runs experiments for all Coherent Ising Machine (CIM) configurations, problem instances, and parameter configs.
"""

import os
import json
from utils.misc import combine_cim_results_to_excel, print_device_info
from bayesian_optimization.optimizer import CIMOptimizer
from simulator import CIMSimulator

cim_architectures = ["cac", "cfc", "snn", "fon", "qa"]

def main():
    print_device_info()

    trials = 100               #number of trials for each problem instance

    for name in cim_architectures:
        print(f"\nRunning experiments for {name} architecture...\n")
        config_path = f"config/config_{name}.json"
        CIM = CIMSimulator(config_path)
        CIM.run_all(trials=trials)

        #CIMSimulator.run_noise_scan(problems, trials, noise_levels)
        #CIMSimulator.run_new_param_scan(problems, trials, new_params)
        #CIMSimulator.run_linear_pump_rate(problems, trials, linear_pump_schedule)

    print("All scheduled experiments have been completed successfully.")
    
if __name__ == "__main__":
    main()