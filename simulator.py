import os
import numpy as np
from utils import misc
from utils import load_graph
import json
from cim_models import standard_cim, cim_qa, cim_snn, cim_cac, cim_cfc, cim_fon
import csv
import pandas as pd

target_gs_energies = [0.87, 0.90, 0.92, 0.96]

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

class CIMSimulator:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = json.load(f)
        
        self.T = self.config["T"]
        self.dt = self.config["dt"]
        self.noise_level = self.config["noise_level"]
        self.cim_type = self.config["cim_type"]
        self.problems = self.config["J_files"]

        self.N = None
        self.J = None
        self.x0 = None
        self.cim_func = None


    def set_problem_instance_and_params(self, problem):
        self.N = problem["N"]
        self.J = load_graph.get_J(problem["path"])
        self.x0 = np.random.uniform(-0.01, 0.01, self.N)

        for key, value in problem.items():          #set optimal parameters for that problem instance
            if key not in {"N", "path"}:
                setattr(self, key, value)

        if self.N > 200:                                                             #threshold for GPU dispatch       
            self.cim_func = CIM_DISPATCH_GPU[self.cim_type]
        else:
            self.cim_func = CIM_DISPATCH[self.cim_type]


    def run_all(self, trials=1):
        for problem in self.problems:
            print(f"Running trials for problem: {problem['path']} with N={problem['N']}")
            self.set_problem_instance_and_params(problem)           #set problem instance

            fixed_keys = {"N", "path"}
            optimal_params = {k: v for k, v in problem.items() if k not in fixed_keys}      #set parameters for that problem

            all_trials = []     #store results of all trials for that problem

            for trial in range(trials):
                self.x0 = np.random.uniform(-0.01, 0.01, self.N)

                states, x, simulation_time = self.cim_func(
                    self.x0,
                    J=self.J,
                    noise_level=self.noise_level,
                    dt=self.dt,
                    T=self.T,
                    N=self.N,
                    **optimal_params
                )

                partition = misc.create_partition_from_opo_amplitudes(x)
                final_cut_val = misc.evaluate_maxcut_fast(self.J, partition)

                all_trials.append({"states": states, "final_cut_val": final_cut_val, "simulation_time": simulation_time})

            #avg_final_cut_val = np.mean([final_cut_val for _, final_cut_val, _ in all_trials])
            #avg_simulation_time = np.mean([simulation_time for _, _, simulation_time in all_trials])

            self.process_and_log_baseline_results(problem, all_trials)

        self.save_to_excel()


    def process_and_log_baseline_results(self, problem, all_trials, experiment_type='baseline'):
        cim_type = self.cim_type

        results_dir = os.path.join("experimental_results", experiment_type, cim_type)
        os.makedirs(results_dir, exist_ok=True)

        problem_name = problem["path"].split("/")[-1].replace(".txt", "")       # Extracting the problem name from the path
        filename = os.path.join(results_dir, f"{problem_name}_{experiment_type}_results.csv")

        optimal_cut_value = self.load_optimal_cut(problem_name)  #load optimal cut value from metadata

        all_trials_gs_times = []  # Store times to reach each target GS energy
        for i, trial in enumerate(all_trials):
            target_gs_times = []
            for target_gs_energy in target_gs_energies:
                time_to_gs_energy = self.get_cut_for_gs_energy(optimal_cut_value,target_gs_energy, trial['states'])
                target_gs_times.append(time_to_gs_energy)
            
            all_trials_gs_times.append(target_gs_times)

        #writing to csv file
        with open(filename, mode='w', newline='') as csvfile:

            fieldnames = ['Trial', 'Cut at T=100', 'Simulation Time (s)', f'T to {target_gs_energies[0]} GS Energy', f'T to {target_gs_energies[1]} GS Energy', f'T to {target_gs_energies[2]} GS Energy', f'T to {target_gs_energies[3]} GS Energy', f'Success for {target_gs_energies[0]} GS Energy', f'Success for {target_gs_energies[1]} GS Energy', f'Success for {target_gs_energies[2]} GS Energy', f'Success for {target_gs_energies[3]} GS Energy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, trial in enumerate(all_trials):
                final_max = trial.get('final_cut_val')  # Adjust key if needed
                sim_time = trial.get('simulation_time')  # Adjust key if needed
                writer.writerow({
                    'Trial': i + 1,
                    'Cut at T=100': final_max,
                    'Simulation Time (s)': sim_time,
                    f'Success for {target_gs_energies[0]} GS Energy': 1 if all_trials_gs_times[i][0] is not None else 0,
                    f'Success for {target_gs_energies[1]} GS Energy': 1 if all_trials_gs_times[i][1] is not None else 0,
                    f'Success for {target_gs_energies[2]} GS Energy': 1 if all_trials_gs_times[i][2] is not None else 0,
                    f'Success for {target_gs_energies[3]} GS Energy': 1 if all_trials_gs_times[i][3] is not None else 0,
                    f'T to {target_gs_energies[0]} GS Energy': all_trials_gs_times[i][0] if all_trials_gs_times[i][0] is not None else self.T,
                    f'T to {target_gs_energies[1]} GS Energy': all_trials_gs_times[i][1] if all_trials_gs_times[i][1] is not None else self.T,
                    f'T to {target_gs_energies[2]} GS Energy': all_trials_gs_times[i][2] if all_trials_gs_times[i][2] is not None else self.T,
                    f'T to {target_gs_energies[3]} GS Energy': all_trials_gs_times[i][3] if all_trials_gs_times[i][3] is not None else self.T,
                })

    def get_cut_for_gs_energy(self, optimal_cut_val, target_gs_energy, states):
        target = target_gs_energy * optimal_cut_val
        best_time = None
        coarse_stride = 10
        step_save_interval = int((self.T / self.dt) / (len(states) - 1))

        #phase 1: coarse scan
        for i in range(0, len(states), coarse_stride):
            partition = misc.create_partition_from_opo_amplitudes(states[i])
            cut_val = misc.evaluate_maxcut_fast(self.J, partition)
            if cut_val >= target:
                #phase 2: fine scan around coarse match
                start = max(0, i - coarse_stride)
                for j in range(start, i + 1):
                    partition = misc.create_partition_from_opo_amplitudes(states[j])
                    cut_val = misc.evaluate_maxcut_fast(self.J, partition)
                    if cut_val >= target:
                        return j * self.dt * step_save_interval  # return earliest time found
                break
        return None  # if not found

    def load_optimal_cut(self, problem_name):
        with open("data/problem_metadata.json", "r") as f:
            metadata = json.load(f)
        return metadata[problem_name]["optimal_cut_value"]

    
    def save_to_excel(self):
        results_dir = os.path.join("experimental_results", "baseline", self.cim_type)
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join("experimental_results", "baseline", f"baseline_results_{self.cim_type}.xlsx")

        with pd.ExcelWriter(output_file) as writer:
            for filename in os.listdir(results_dir):
                if filename.endswith(".csv"):
                    file_path = os.path.join(results_dir, filename)
                    sheet_name = os.path.splitext(filename)[0][:31]  # Excel sheet name limit = 31 characters

                    try:
                        df = pd.read_csv(file_path)
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")


    def run_noise_scan(self, trials=1, noise_levels=None, select_problems=None):
        for problem in self.problems:
            if select_problems and problem["path"] not in select_problems:
                continue
            self.set_problem_instance_and_params(problem)           #set problem instance

            for noise_level in noise_levels:
                self.noise_level = noise_level  # Set the noise level for this run
                print(f"Running trials for problem: {problem['path']} with N={problem['N']} and noise level={noise_level}")

                fixed_keys = {"N", "path"}
                optimal_params = {k: v for k, v in problem.items() if k not in fixed_keys}      #set parameters for that problem

                all_trials = []     #store results of all trials for that problem

                for trial in range(trials):
                    self.x0 = np.random.uniform(-0.01, 0.01, self.N)

                    states, x, simulation_time = self.cim_func(
                        self.x0,
                        J=self.J,
                        noise_level=self.noise_level,
                        dt=self.dt,
                        T=self.T,
                        N=self.N,
                        **optimal_params
                    )

                    partition = misc.create_partition_from_opo_amplitudes(x)
                    final_cut_val = misc.evaluate_maxcut_fast(self.J, partition)

                    all_trials.append({"states": states, "final_cut_val": final_cut_val})
                    #print(f"Trial {trial + 1}/{trials} completed. Cut value: {final_cut_val}")

                avg_final_cut_val = np.mean([trial["final_cut_val"] for trial in all_trials])

                #print(f"Average final cut value: {avg_final_cut_val}")

                self.process_and_log_noise_scan_results(problem, all_trials, noise_level)


    def process_and_log_noise_scan_results(self, problem, all_trials, noise_level, experiment_type='noise_scan'):
        cim_type = self.cim_type

        problem_name = problem["path"].split("/")[-1].replace(".txt", "")       # Extracting the problem name from the path

        results_dir = os.path.join("experimental_results", experiment_type, cim_type, problem_name)
        os.makedirs(results_dir, exist_ok=True)

        filename = os.path.join(results_dir, f"{problem_name}_noise_{noise_level}_results.csv")

        optimal_cut_value = self.load_optimal_cut(problem_name)  #load optimal cut value from metadata

        all_trials_gs_times = []  # Store times to reach each target GS energy
        for i, trial in enumerate(all_trials):
            target_gs_times = []
            for target_gs_energy in target_gs_energies:
                time_to_gs_energy = self.get_cut_for_gs_energy(optimal_cut_value,target_gs_energy, trial['states'])
                target_gs_times.append(time_to_gs_energy)
            
            all_trials_gs_times.append(target_gs_times)

        #writing to csv file
        with open(filename, mode='w', newline='') as csvfile:
            fieldnames = [
                "Trial",
                "Noise Level",
                "Cut at T=100",
            ]
            for energy in target_gs_energies:
                fieldnames.append(f"T to {energy} GS Energy")
            for energy in target_gs_energies:
                fieldnames.append(f"Success for {energy} GS Energy")

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, trial in enumerate(all_trials):
                final_max = trial.get('final_cut_val')
                writer.writerow({
                    'Trial': i + 1,
                    'Cut at T=100': final_max,
                    'Noise Level': noise_level,
                    f'Success for {target_gs_energies[0]} GS Energy': 1 if all_trials_gs_times[i][0] is not None else 0,
                    f'Success for {target_gs_energies[1]} GS Energy': 1 if all_trials_gs_times[i][1] is not None else 0,
                    f'Success for {target_gs_energies[2]} GS Energy': 1 if all_trials_gs_times[i][2] is not None else 0,
                    f'Success for {target_gs_energies[3]} GS Energy': 1 if all_trials_gs_times[i][3] is not None else 0,
                    f'T to {target_gs_energies[0]} GS Energy': all_trials_gs_times[i][0] if all_trials_gs_times[i][0] is not None else self.T,
                    f'T to {target_gs_energies[1]} GS Energy': all_trials_gs_times[i][1] if all_trials_gs_times[i][1] is not None else self.T,
                    f'T to {target_gs_energies[2]} GS Energy': all_trials_gs_times[i][2] if all_trials_gs_times[i][2] is not None else self.T,
                    f'T to {target_gs_energies[3]} GS Energy': all_trials_gs_times[i][3] if all_trials_gs_times[i][3] is not None else self.T,
                })

    def run_constant_pump_scan(self, trials=1, pump_rates=None, select_problems=None):
        for problem in self.problems:
            if select_problems and problem["path"] not in select_problems:
                continue
            self.set_problem_instance_and_params(problem)           #set problem instance

            for pump_rate in pump_rates:
                self.p = pump_rate  # Set the pump rate for this run
                print(f"Running trials for problem: {problem['path']} with N={problem['N']} and pump rate={self.p}")

                fixed_keys = {"N", "path", "p"}
                optimal_params = {k: v for k, v in problem.items() if k not in fixed_keys}      #set parameters for that problem

                all_trials = []     #store results of all trials for that problem

                for trial in range(trials):
                    self.x0 = np.random.uniform(-0.01, 0.01, self.N)

                    states, x, simulation_time = self.cim_func(
                        self.x0,
                        J=self.J,
                        noise_level=self.noise_level,
                        dt=self.dt,
                        T=self.T,
                        N=self.N,
                        p=self.p,
                        **optimal_params
                    )

                    partition = misc.create_partition_from_opo_amplitudes(x)
                    final_cut_val = misc.evaluate_maxcut_fast(self.J, partition)

                    all_trials.append({"states": states, "final_cut_val": final_cut_val})
                    #print(f"Trial {trial + 1}/{trials} completed. Cut value: {final_cut_val}")

                avg_final_cut_val = np.mean([trial["final_cut_val"] for trial in all_trials])

                #print(f"Average final cut value: {avg_final_cut_val}")

                self.process_and_log_pump_scan_results(problem, all_trials, pump_rate)

    def process_and_log_pump_scan_results(self, problem, all_trials, pump_rate, experiment_type='constant_pump_scan'):
        cim_type = self.cim_type

        problem_name = problem["path"].split("/")[-1].replace(".txt", "")       # Extracting the problem name from the path

        results_dir = os.path.join("experimental_results", experiment_type, cim_type, problem_name)
        os.makedirs(results_dir, exist_ok=True)

        filename = os.path.join(results_dir, f"{problem_name}_constant_pump_{pump_rate}_results.csv")

        optimal_cut_value = self.load_optimal_cut(problem_name)  #load optimal cut value from metadata

        all_trials_gs_times = []  # Store times to reach each target GS energy
        for i, trial in enumerate(all_trials):
            target_gs_times = []
            for target_gs_energy in target_gs_energies:
                time_to_gs_energy = self.get_cut_for_gs_energy(optimal_cut_value,target_gs_energy, trial['states'])
                target_gs_times.append(time_to_gs_energy)
            
            all_trials_gs_times.append(target_gs_times)

        #writing to csv file
        with open(filename, mode='w', newline='') as csvfile:
            fieldnames = [
                "Trial",
                "Pump Rate",
                "Cut at T=100",
            ]
            for energy in target_gs_energies:
                fieldnames.append(f"T to {energy} GS Energy")
            for energy in target_gs_energies:
                fieldnames.append(f"Success for {energy} GS Energy")

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, trial in enumerate(all_trials):
                final_max = trial.get('final_cut_val')
                writer.writerow({
                    'Trial': i + 1,
                    'Cut at T=100': final_max,
                    'Pump Rate': pump_rate,
                    f'Success for {target_gs_energies[0]} GS Energy': 1 if all_trials_gs_times[i][0] is not None else 0,
                    f'Success for {target_gs_energies[1]} GS Energy': 1 if all_trials_gs_times[i][1] is not None else 0,
                    f'Success for {target_gs_energies[2]} GS Energy': 1 if all_trials_gs_times[i][2] is not None else 0,
                    f'Success for {target_gs_energies[3]} GS Energy': 1 if all_trials_gs_times[i][3] is not None else 0,
                    f'T to {target_gs_energies[0]} GS Energy': all_trials_gs_times[i][0] if all_trials_gs_times[i][0] is not None else self.T,
                    f'T to {target_gs_energies[1]} GS Energy': all_trials_gs_times[i][1] if all_trials_gs_times[i][1] is not None else self.T,
                    f'T to {target_gs_energies[2]} GS Energy': all_trials_gs_times[i][2] if all_trials_gs_times[i][2] is not None else self.T,
                    f'T to {target_gs_energies[3]} GS Energy': all_trials_gs_times[i][3] if all_trials_gs_times[i][3] is not None else self.T,
                })
        

    def run_linear_pump_rate(self, trials=1, linear_pump_schedule=None):
        for problem in self.problems:
            print(f"Running trials for problem: {problem['path']} with N={problem['N']}")
            self.set_problem_instance_and_params(problem)           #set problem instance

            fixed_keys = {"N", "path"}
            optimal_params = {k: v for k, v in problem.items() if k not in fixed_keys}      #set parameters for that problem

            all_trials = []     #store results of all trials for that problem

            for trial in range(trials):
                self.x0 = np.random.uniform(-0.01, 0.01, self.N)

                states, x, simulation_time = self.cim_func(
                    self.x0,
                    J=self.J,
                    noise_level=self.noise_level,
                    dt=self.dt,
                    T=self.T,
                    N=self.N,
                    **optimal_params,
                    linear_pump_schedule=linear_pump_schedule
                )

                partition = misc.create_partition_from_opo_amplitudes(x)
                final_cut_val = misc.evaluate_maxcut_fast(self.J, partition)

                all_trials.append({"states": states, "final_cut_val": final_cut_val, "simulation_time": simulation_time})


            self.process_and_log_linear_pump_results(problem, all_trials)

    

    def process_and_log_linear_pump_results(self, problem, all_trials, experiment_type='linear_pump_rate'):
        cim_type = self.cim_type

        results_dir = os.path.join("experimental_results", experiment_type, cim_type)
        os.makedirs(results_dir, exist_ok=True)

        problem_name = problem["path"].split("/")[-1].replace(".txt", "")       # Extracting the problem name from the path
        filename = os.path.join(results_dir, f"{problem_name}_{experiment_type}_results.csv")

        optimal_cut_value = self.load_optimal_cut(problem_name)  #load optimal cut value from metadata

        all_trials_gs_times = []  # Store times to reach each target GS energy
        for i, trial in enumerate(all_trials):
            target_gs_times = []
            for target_gs_energy in target_gs_energies:
                time_to_gs_energy = self.get_cut_for_gs_energy(optimal_cut_value,target_gs_energy, trial['states'])
                target_gs_times.append(time_to_gs_energy)
            
            all_trials_gs_times.append(target_gs_times)

        #writing to csv file
        with open(filename, mode='w', newline='') as csvfile:

            fieldnames = ['Trial', 'Cut at T=100', f'T to {target_gs_energies[0]} GS Energy', f'T to {target_gs_energies[1]} GS Energy', f'T to {target_gs_energies[2]} GS Energy', f'T to {target_gs_energies[3]} GS Energy', f'Success for {target_gs_energies[0]} GS Energy', f'Success for {target_gs_energies[1]} GS Energy', f'Success for {target_gs_energies[2]} GS Energy', f'Success for {target_gs_energies[3]} GS Energy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, trial in enumerate(all_trials):
                final_max = trial.get('final_cut_val')  # Adjust key if needed
                writer.writerow({
                    'Trial': i + 1,
                    'Cut at T=100': final_max,
                    f'Success for {target_gs_energies[0]} GS Energy': 1 if all_trials_gs_times[i][0] is not None else 0,
                    f'Success for {target_gs_energies[1]} GS Energy': 1 if all_trials_gs_times[i][1] is not None else 0,
                    f'Success for {target_gs_energies[2]} GS Energy': 1 if all_trials_gs_times[i][2] is not None else 0,
                    f'Success for {target_gs_energies[3]} GS Energy': 1 if all_trials_gs_times[i][3] is not None else 0,
                    f'T to {target_gs_energies[0]} GS Energy': all_trials_gs_times[i][0] if all_trials_gs_times[i][0] is not None else self.T,
                    f'T to {target_gs_energies[1]} GS Energy': all_trials_gs_times[i][1] if all_trials_gs_times[i][1] is not None else self.T,
                    f'T to {target_gs_energies[2]} GS Energy': all_trials_gs_times[i][2] if all_trials_gs_times[i][2] is not None else self.T,
                    f'T to {target_gs_energies[3]} GS Energy': all_trials_gs_times[i][3] if all_trials_gs_times[i][3] is not None else self.T,
                })