import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import os
import psutil
import platform

cim_architectures = ["standard", "snn", "fon", "qa", "cfc", "cac"]

def combine_cim_results_to_excel(cim_type):
    folder_path = os.path.join("..", "results", "optimization_results", cim_type)
    excel_output_path = os.path.join(f"../results/optimization_results/{cim_type}", f"{cim_type}_results.xlsx")
    
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

if __name__ == "__main__":
    for cim_type in cim_architectures:
        print(f"Combining results for CIM type: {cim_type}")
        combine_cim_results_to_excel(cim_type)
        print(f"Finished combining results for CIM type: {cim_type}")