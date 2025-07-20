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

cim_architectures = ["standard", "cac", "cfc", "snn", "fon", "qa"]

def main():
    print_device_info()

    trials = 100               #number of trials for each problem instance

    #noise_levels = [0.0, 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.99]

    #constant_pump_rates = [1.001, 1.005, 1.01, 1.05, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]

    linear_pump_schedule = {"start": 1.0001, "end": 2.0}

    select_problems = [
        "data/biqmac/pm1s/pm1s_80.4.txt",
        "data/biqmac/pm1s/pm1s_80.7.txt",
        "data/biqmac/pm1s/pm1s_100.9.txt",
        "data/biqmac/pm1s/pm1s_100.5.txt",
        "data/biqmac/w01/w01_100.5.txt",
        "data/biqmac/w01/w01_100.8.txt",
        "data/biqmac/w05/w05_100.2.txt",
        "data/biqmac/w05/w05_100.9.txt",
        "data/biqmac/w09/w09_100.0.txt",
        "data/biqmac/w09/w09_100.3.txt",
        "data/gset/G6.txt",
        "data/gset/G9.txt",
        "data/gset/G27.txt",
        "data/gset/G30.txt"
    ]

    for name in cim_architectures:
        print(f"\nRunning experiments for {name} architecture...\n")
        config_path = f"config/config_{name}.json"
        CIM = CIMSimulator(config_path)

        #CIM.run_all(trials=trials)

        #CIM.run_noise_scan(trials=trials, noise_levels=noise_levels, select_problems=select_problems)
        #CIM.run_constant_pump_scan(trials=trials, pump_rates=constant_pump_rates, select_problems=select_problems)

        CIM.run_linear_pump_rate(trials=trials, linear_pump_schedule=linear_pump_schedule)

    print("\nAll scheduled experiments have been completed successfully.")
    
if __name__ == "__main__":
    main()