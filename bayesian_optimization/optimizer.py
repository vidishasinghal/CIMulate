import os
import numpy as np
from bayes_opt import BayesianOptimization
from utils import misc
from utils import load_graph
import json
from cim_models import standard_cim, cim_qa, cim_snn, cim_cac, cim_cfc, cim_fon
import concurrent.futures

CIM_DISPATCH = {
    "standard": standard_cim.standard_cim,
    "snn": cim_snn.cim_snn,
    "qa": cim_qa.cim_qa,
    "cac": cim_cac.cim_cac,
    "cfc": cim_cfc.cim_cfc,
    "fon": cim_fon.cim_fon
}

CIM_DISPATCH_GPU = {
    "standard": standard_cim.standard_cim_gpu,
    "snn": cim_snn.cim_snn_gpu,
    "qa": cim_qa.cim_qa_gpu,
    "cac": cim_cac.cim_cac_gpu,
    "cfc": cim_cfc.cim_cfc_gpu,
    "fon": cim_fon.cim_fon_gpu
}

class CIMOptimizer:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = json.load(f)
        
        self.T = self.config["T"]
        self.dt = self.config["dt"]
        self.noise_level = self.config["noise_level"]
        self.cim_type = self.config["cim_type"]
        self.pbounds = self.config["pbounds"]
        self.problems = self.config["J_files"]

        self.N = None
        self.J = None
        self.x0 = None
        self.cim_func = None

        self.output_file = self.config["output_file"]
    
    def set_problem_instance(self, problem):
        self.N = problem["N"]
        self.J = load_graph.get_J(problem["path"])
        self.x0 = np.random.uniform(-0.01, 0.01, self.N)

        if self.N > 200:                                                             #threshold for GPU dispatch       
            self.cim_func = CIM_DISPATCH_GPU[self.cim_type]
        else:
            self.cim_func = CIM_DISPATCH[self.cim_type]


    def black_box_function(self, **params):
        if self.N is None or self.J is None or self.x0 is None:
            raise ValueError("Problem instance not set. Call set_problem_instance() first.")
        
        self.x0 = np.random.uniform(-0.01, 0.01, self.N)                            # Reset initial state for each run of optimization

        states, x = self.cim_func(
            self.x0,
            J=self.J,
            noise_level=self.noise_level,
            dt=self.dt,
            T=self.T,
            N=self.N,
            **params
        )

        partition = misc.create_partition_from_opo_amplitudes(x)
        cut_val = misc.evaluate_maxcut_fast(self.J, partition)

        return cut_val


    def optimize_all(self, init_points=10, n_iter=10, trials=1):
        for problem in self.problems:
            self.set_problem_instance(problem)
            print(f"Optimizing problem: {problem['path']}")

            for trial in range(trials):
                print(f"Trial {trial + 1}/{trials} for problem {problem['path']}")

                optimizer = BayesianOptimization(
                    f=self.black_box_function, 
                    pbounds=self.pbounds,
                    random_state=np.random.seed(trial)
                )

                optimizer.maximize(
                    init_points=init_points if init_points is not None else 10,
                    n_iter=n_iter if n_iter is not None else 10
                )

                params = optimizer.max["params"]
                target = optimizer.max["target"]

                print("Best result: {}; f(x) = {:.3f}.".format(params, target))

                self.log_results(params, target, problem, trial)


    def optimize_all_multithreaded(self, init_points=10, n_iter=10, trials=1):
        for problem in self.problems:
            self.set_problem_instance(problem)
            print(f"Optimizing problem: {problem['path']}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=trials) as executor:
                futures = []

                for trial in range(trials):
                    print(f"Scheduling Trial {trial + 1}/{trials} for problem {problem['path']}")
                    futures.append(executor.submit(lambda t=trial: self.run_single_trial(t, problem, init_points, n_iter)))
                
                for future in concurrent.futures.as_completed(futures):
                    future.result()

    
    def run_single_trial(self, trial, problem, init_points, n_iter):
        
        '''for use with optimize_all_multithreaded'''

        optimizer = BayesianOptimization(
            f=self.black_box_function, 
            pbounds=self.pbounds,
            #random_state=1
        )

        optimizer.maximize(
            init_points=init_points if init_points is not None else 10,
            n_iter=n_iter if n_iter is not None else 10
        )

        params = optimizer.max["params"]
        target = optimizer.max["target"]

        print("Best result: {}; f(x) = {:.3f}.".format(params, target))

        self.log_results(params, target, problem, trial)



    def log_results(self, max_params, max_target, problem, trial):
        cim_type = self.cim_type  # e.g., 'standard', 'snn', etc.
        results_dir = os.path.join("results", "optimization_results", cim_type)
        os.makedirs(results_dir, exist_ok=True)

        problem_name = os.path.splitext(os.path.basename(problem["path"]))[0]
        filename = os.path.join(results_dir, f"{problem_name}_optimization_results.json")

        entry = {
            "trial": trial + 1,
            "optimal_params": max_params,
            "maxcut_value": max_target
        }

        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(entry)

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)