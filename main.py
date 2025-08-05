# main.py
# This is the main script to run the SVM experiments.

import numpy as np
from sklearn.model_selection import train_test_split

# Import our custom modules
import data_utils
from frank_wolfe_svm import FrankWolfeOptimizer
from pgd_svm import PGD
import visualizations

def run_experiments():
    """
    Main function to run all experiments.
    """
    # --- Load and Prepare Datasets ---
    print("=== Loading Data ===")
    X_a4a, y_a4a = data_utils.load_a4a_dataset()
    X_bc, y_bc = data_utils.load_and_prepare_breast_cancer_data()

    # Split datasets
    X_train_a4a, X_test_a4a, y_train_a4a, y_test_a4a = train_test_split(X_a4a, y_a4a, test_size=0.25, random_state=42)
    X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.25, random_state=42)
    
    # Normalize the breast cancer dataset
    X_train_bc_norm, X_test_bc_norm = data_utils.check_and_normalize(X_train_bc, X_test_bc)

    # Check class balance
    data_utils.check_class_balance(y_a4a, "a4a Full Dataset")
    data_utils.check_class_balance(y_bc, "Breast Cancer Full Dataset")

    results = []

    # --- Frank-Wolfe Experiments ---
    print("\n=== Running Frank-Wolfe Experiments ===")
    
    # Breast Cancer - Diminishing
    print("\nTraining FW on Breast Cancer (Diminishing)...")
    fw_bc_dim = FrankWolfeOptimizer(C=1, epsilon=1e-6, max_iterations=3000).fit(X_train_bc_norm, y_train_bc, step_size_strategy="diminishing")
    y_pred_fw_bc_dim = fw_bc_dim.predict(X_test_bc_norm)
    acc_fw_bc_dim = fw_bc_dim.compute_accuracy(y_test_bc, y_pred_fw_bc_dim)
    print(f"Accuracy: {acc_fw_bc_dim*100:.2f}%")
    results.append({
        "Model Name": "FW", "Dataset": "Breast Cancer", "Line Search": "Diminishing",
        "Final Objective Value": fw_bc_dim.history["objective_value"][-1],
        "Accuracy (%)": acc_fw_bc_dim * 100,
        "Total CPU Time (s)": fw_bc_dim.history["cpu_time"][-1]
    })

    # Breast Cancer - Armijo
    print("\nTraining FW on Breast Cancer (Armijo)...")
    fw_bc_armijo = FrankWolfeOptimizer(C=1, epsilon=1e-6, max_iterations=2000).fit(X_train_bc_norm, y_train_bc, step_size_strategy="armijo")
    y_pred_fw_bc_armijo = fw_bc_armijo.predict(X_test_bc_norm)
    acc_fw_bc_armijo = fw_bc_armijo.compute_accuracy(y_test_bc, y_pred_fw_bc_armijo)
    print(f"Accuracy: {acc_fw_bc_armijo*100:.2f}%")
    results.append({
        "Model Name": "FW", "Dataset": "Breast Cancer", "Line Search": "Armijo",
        "Final Objective Value": fw_bc_armijo.history["objective_value"][-1],
        "Accuracy (%)": acc_fw_bc_armijo * 100,
        "Total CPU Time (s)": fw_bc_armijo.history["cpu_time"][-1]
    })

    # a4a - Diminishing
    print("\nTraining FW on a4a (Diminishing)...")
    fw_a4a_dim = FrankWolfeOptimizer(C=0.1, epsilon=1e-6, max_iterations=5000).fit(X_train_a4a, y_train_a4a, step_size_strategy="diminishing")
    y_pred_fw_a4a_dim = fw_a4a_dim.predict(X_test_a4a)
    acc_fw_a4a_dim = fw_a4a_dim.compute_accuracy(y_test_a4a, y_pred_fw_a4a_dim)
    print(f"Accuracy: {acc_fw_a4a_dim*100:.2f}%")
    results.append({
        "Model Name": "FW", "Dataset": "a4a", "Line Search": "Diminishing",
        "Final Objective Value": fw_a4a_dim.history["objective_value"][-1],
        "Accuracy (%)": acc_fw_a4a_dim * 100,
        "Total CPU Time (s)": fw_a4a_dim.history["cpu_time"][-1]
    })

    # a4a - Armijo
    print("\nTraining FW on a4a (Armijo)...")
    fw_a4a_armijo = FrankWolfeOptimizer(C=0.1, epsilon=1e-6, max_iterations=5000).fit(X_train_a4a, y_train_a4a, step_size_strategy="armijo")
    y_pred_fw_a4a_armijo = fw_a4a_armijo.predict(X_test_a4a)
    acc_fw_a4a_armijo = fw_a4a_armijo.compute_accuracy(y_test_a4a, y_pred_fw_a4a_armijo)
    print(f"Accuracy: {acc_fw_a4a_armijo*100:.2f}%")
    results.append({
        "Model Name": "FW", "Dataset": "a4a", "Line Search": "Armijo",
        "Final Objective Value": fw_a4a_armijo.history["objective_value"][-1],
        "Accuracy (%)": acc_fw_a4a_armijo * 100,
        "Total CPU Time (s)": fw_a4a_armijo.history["cpu_time"][-1]
    })

    # --- PGD Experiments ---
    print("\n=== Running Projected Gradient Descent Experiments ===")
    
    # Breast Cancer - Armijo
    print("\nTraining PGD on Breast Cancer (Armijo)...")
    pgd_bc_armijo = PGD(C=0.3, epsilon=1e-6, s=0.1, max_iterations=2000).fit(X_train_bc_norm, y_train_bc)
    y_pred_pgd_bc = pgd_bc_armijo.predict(X_test_bc_norm)
    acc_pgd_bc = pgd_bc_armijo.compute_accuracy(y_test_bc, y_pred_pgd_bc)
    print(f"Accuracy: {acc_pgd_bc*100:.2f}%")
    results.append({
        "Model Name": "PGD", "Dataset": "Breast Cancer", "Line Search": "Armijo",
        "Final Objective Value": pgd_bc_armijo.history["objective_value"][-1],
        "Accuracy (%)": acc_pgd_bc * 100,
        "Total CPU Time (s)": pgd_bc_armijo.history["cpu_time"][-1]
    })

    # a4a - Armijo
    print("\nTraining PGD on a4a (Armijo)...")
    pgd_a4a_armijo = PGD(C=0.4, epsilon=1e-6, s=0.1, max_iterations=5000).fit(X_train_a4a, y_train_a4a)
    y_pred_pgd_a4a = pgd_a4a_armijo.predict(X_test_a4a)
    acc_pgd_a4a = pgd_a4a_armijo.compute_accuracy(y_test_a4a, y_pred_pgd_a4a)
    print(f"Accuracy: {acc_pgd_a4a*100:.2f}%")
    results.append({
        "Model Name": "PGD", "Dataset": "a4a", "Line Search": "Armijo",
        "Final Objective Value": pgd_a4a_armijo.history["objective_value"][-1],
        "Accuracy (%)": acc_pgd_a4a * 100,
        "Total CPU Time (s)": pgd_a4a_armijo.history["cpu_time"][-1]
    })

    # --- Generate Visualizations ---
    print("\n=== Generating Plots ===")
    visualizations.plot_objective_comparison(
        [fw_a4a_armijo.history, fw_a4a_dim.history, fw_bc_armijo.history, fw_bc_dim.history],
        ["a4a (Armijo FW)", "a4a (Diminishing FW)", "Breast Cancer (Armijo FW)", "Breast Cancer (Diminishing FW)"],
        "Frank-Wolfe: Objective Function vs Iterations"
    )
    visualizations.plot_duality_gap_comparison(
        [fw_a4a_armijo.history, fw_a4a_dim.history, fw_bc_armijo.history, fw_bc_dim.history],
        ["a4a (Armijo)", "a4a (Diminishing)", "Breast Cancer (Armijo)", "Breast Cancer (Diminishing)"],
        "Frank-Wolfe: Duality Gap vs Iterations"
    )
    visualizations.plot_objective_comparison(
        [pgd_a4a_armijo.history, pgd_bc_armijo.history],
        ["a4a (Armijo PGD)", "Breast Cancer (Armijo PGD)"],
        "PGD: Objective Function vs Iterations"
    )
    
    # --- Print Summary ---
    visualizations.print_summary_table(results)


if __name__ == "__main__":
    run_experiments()
