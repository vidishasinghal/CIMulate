import matplotlib.pyplot as plt
import numpy as np
import json
import os
import csv
from statistics import mean, variance
from collections import defaultdict

'''NOTES:
- This script has lots of issues. Optimal parameters are not being averaged correctly since a lot of the trials have a very low maxcut value. Fix this after selecting the best parameters."
'''

cim_architectures = ["standard", "snn", "fon", "qa", "cfc", "cac"]

problems_metadata_path = '../data/problem_metadata.json'

colors = [
    '#7AA9D9',  # pastel red
    '#F28C7B',  # pastel green
    '#9DC77D',  # pastel blue
    '#A388D1',  # pastel purple
    '#F0C987',  # pastel magenta
]

def plot_optimal_param_mean_variance_vs_N():
    #load problem metadata
    with open(problems_metadata_path) as f:
        problems_metadata = json.load(f)
    
    for csv_file in os.listdir('../results/processed_optimization_results'):
        cim_name = csv_file.replace('parameter_mean_variance_per_problem_', '').replace('.csv', '')
        csv_path = os.path.join('../results/processed_optimization_results', csv_file)

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            all_columns = reader.fieldnames
            param_names = [col for col in all_columns if col != 'Problem']

            #structure to hold means and variances grouped by N
            means_grouped_by_N = {param: defaultdict(list) for param in param_names}
            variances_grouped_by_N = {param: defaultdict(list) for param in param_names}

            for row in reader:
                problem_name = row['Problem']
                N = problems_metadata[problem_name]['N']
                for param in param_names:
                    mean_str, std_str = row[param].split('±')
                    mean_val = float(mean_str.strip())
                    std_val = float(std_str.strip())
                    means_grouped_by_N[param][N].append(mean_val)
                    variances_grouped_by_N[param][N].append(std_val ** 2)

        #aggregate data
        aggregated_means = {param: {} for param in param_names}
        aggregated_variances = {param: {} for param in param_names}

        for param in param_names:
            for N, mean_list in means_grouped_by_N[param].items():
                var_list = variances_grouped_by_N[param][N]
                aggregated_means[param][N] = np.mean(mean_list)
                aggregated_variances[param][N] = np.mean(var_list)

        #plotting
        plt.figure(figsize=(12, 8))
        for i, param in enumerate(param_names):
            Ns = sorted(aggregated_means[param].keys())
            means = [aggregated_means[param][N] for N in Ns]
            vars_ = [aggregated_variances[param][N] for N in Ns]
            std_devs = np.sqrt(vars_)

            plt.errorbar(Ns, means, yerr=std_devs, label=param, marker='o', capsize=15, color=colors[i], lw=3, elinewidth=1.2, markersize=8)

        plt.xscale('log') 
        ax = plt.gca()
        plt.xlabel('Problem Size (N)', fontweight='bold')
        plt.ylabel('Parameter Mean ± Std Dev (%)', fontweight='bold')
        plt.title(f'Optimal Parameter Means vs Problem Size for CIM-{cim_name.upper()}', fontsize=15, fontweight='bold')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=len(param_names), fontsize=12)
        plt.grid(True)
        plt.savefig(f'../results/optimization_plots/parameter_vs_N_plot_{cim_name}.png', dpi=300, bbox_inches='tight')
        plt.close()


def calculate_params_mean_and_variance_for_a_problem(results_folder_path):
    stats_across_trials = {}
    for file in os.listdir(results_folder_path):
        if file.endswith(".json"):
            file_path = os.path.join(results_folder_path, file)
            with open(file_path, 'r') as f:
                problem_name = file.replace('_optimization_results.json', '')  #extract problem name from filename
                data = json.load(f)
                params = list(data[0]['optimal_params'].keys())
                param_values = {param: [] for param in params}
                for trial in data:
                    for param in params:
                        param_values[param].append(trial['optimal_params'][param])
                aggregated_stats = {}
                for param in params:
                    vals = param_values[param]
                    avg = mean(vals)
                    var = variance(vals) if len(vals) > 1 else 0.0
                    aggregated_stats[param] = {'mean': avg, 'variance': var}
                stats_across_trials[problem_name] = aggregated_stats
    return stats_across_trials


def save_stats_to_csv(stats_across_trials, cim_name):
    csv_file_path = f'../results/processed_optimization_results/parameter_mean_variance_per_problem_{cim_name}.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        #write header
        header = ['Problem'] + list(next(iter(stats_across_trials.values())).keys())
        writer.writerow(header)
        #write data
        for filename, stats in stats_across_trials.items():
            row = [filename] + [f"{stats[param]['mean']} ± {np.sqrt(stats[param]['variance'])}" for param in header[1:]]
            writer.writerow(row)
    print(f"Stats saved to {csv_file_path}")


def main():
    for name in cim_architectures:
        results_folder_path = f"../results/optimization_results/{name}"
        stats_across_trials = calculate_params_mean_and_variance_for_a_problem(results_folder_path)    #calculate stats across trials for each CIM architecture separately (all problems)
        save_stats_to_csv(stats_across_trials, name)                                                  #save stats to csv file for each CIM architecture (all problems)
   
    plot_optimal_param_mean_variance_vs_N()                     #plot for all CIM architectures
    print("Finished plotting!")

if __name__ == "__main__":
    main()