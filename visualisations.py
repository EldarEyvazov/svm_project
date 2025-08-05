# visualizations.py
# This file contains functions for plotting the results of the model training.

import matplotlib.pyplot as plt
import prettytable

def plot_objective_comparison(histories, labels, title):
    """
    Plots a comparison of objective function values from multiple training histories.
    """
    plt.figure(figsize=(12, 6))
    for history, label in zip(histories, labels):
        plt.plot(history["iteration"], history["objective_value"], linestyle='-', label=label)
    
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value (Log Scale)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

def plot_duality_gap_comparison(histories, labels, title):
    """
    Plots a comparison of duality gaps from multiple Frank-Wolfe training histories.
    """
    plt.figure(figsize=(12, 6))
    for history, label in zip(histories, labels):
        # Align lengths of iterations and dual_gap
        iterations = history["iteration"]
        dual_gaps = history["dual_gap"]
        min_len = min(len(iterations), len(dual_gaps))
        plt.semilogy(iterations[:min_len], dual_gaps[:min_len], linestyle='-', label=label)

    plt.xlabel('Iteration')
    plt.ylabel('Duality Gap (Log Scale)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

def plot_cpu_time_comparison(histories, labels, title):
    """
    Plots a comparison of CPU times from multiple training histories.
    """
    plt.figure(figsize=(12, 6))
    for history, label in zip(histories, labels):
        plt.plot(history["iteration"], history["cpu_time"], linestyle='-', label=label)
    
    plt.xlabel('Iteration')
    plt.ylabel('CPU Time (seconds)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

def plot_gradient_mapping_comparison(histories, labels, title):
    """
    Plots a comparison of gradient mapping norms from multiple PGD training histories.
    """
    plt.figure(figsize=(12, 6))
    for history, label in zip(histories, labels):
        iterations = history["iteration"]
        gradient_mappings = history["gradient_mapping"]
        min_len = min(len(iterations), len(gradient_mappings))
        plt.semilogy(iterations[:min_len], gradient_mappings[:min_len], linestyle='-', label=label)
        
    plt.xlabel("Iterations")
    plt.ylabel("Gradient Mapping (Log Scale)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

def print_summary_table(results):
    """
    Prints a summary of training results in a formatted table.
    'results' should be a list of dictionaries, each containing the keys:
    'Model Name', 'Dataset', 'Line Search', 'Final Objective Value', 
    'Accuracy (%)', 'Total CPU Time (s)'
    """
    table = prettytable.PrettyTable()
    table.field_names = [
        "Model Name", "Dataset", "Line Search",
        "Final Objective Value", "Accuracy (%)",
        "Total CPU Time (s)"
    ]
    
    for result in results:
        table.add_row([
            result["Model Name"],
            result["Dataset"],
            result["Line Search"],
            f"{result['Final Objective Value']:.4e}",
            f"{result['Accuracy (%)']:.2f}",
            f"{result['Total CPU Time (s)']:.2f}"
        ])
        
    print("\n--- Training Results Summary ---")
    print(table)
