# SVM Implementation with Frank-Wolfe and Projected Gradient Descent

This project provides a Python implementation of Support Vector Machines (SVM) using two different optimization algorithms: the Frank-Wolfe (FW) algorithm and Projected Gradient Descent (PGD).

The models are trained and evaluated on two public datasets:
-   The **a4a** dataset (sparse)
-   The **Breast Cancer** dataset (dense)

## Project Structure

-   `main.py`: The main script to run all experiments and generate plots.
-   `data_utils.py`: Contains functions for downloading, loading, and preprocessing the datasets.
-   `frank_wolfe_svm.py`: Contains the class implementation for the Frank-Wolfe SVM optimizer.
-   `pgd_svm.py`: Contains the class implementation for the Projected Gradient Descent SVM optimizer.
-   `visualizations.py`: Contains functions for plotting the results and generating summary tables.
-   `requirements.txt`: Lists the necessary Python libraries to run the project.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Main Script:**
    ```bash
    python main.py
    ```

The script will automatically download the datasets, train the models, and display the comparative plots for objective function value, duality gap, and CPU time.

