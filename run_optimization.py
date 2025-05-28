import os
import json
from utils.misc import combine_cim_results_to_excel
from bayesian_optimization.optimizer import CIMOptimizer

config_names = ["standard"] #, "snn", "qa", "cac", "cfc", "fon"]

def main():
    init_points = 10
    n_iter = 10
    trials = 5

    config_files = [f"config/config_optimizations/config_{name}.json" for name in config_names]

    for config_path in config_files:

        CIM = CIMOptimizer(config_path)       
        CIMOptimizer.optimize_all(CIM, trials=trials)
    
    print("I have successfully called optimize on the test problem!")

    combine_cim_results_to_excel("standard")


if __name__ == "__main__":
    main()