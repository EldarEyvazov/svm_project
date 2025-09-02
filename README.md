# Optimization Methods for SVM Training

## Overview
This project investigates **optimization techniques for training soft-margin Support Vector Machines (SVMs)** using:
- **Frank–Wolfe algorithm** (projection-free)
- **Projected Gradient Descent (PGD)** (projection-based)

The focus is on the **dual formulation of SVMs** and the trade-offs between projection-free and projection-based methods in terms of **computational cost, convergence, and accuracy**.  
Two real-world datasets are used for evaluation: **a4a** (binary, high-dimensional) and **Breast Cancer** (binary, low-dimensional).

## Problem Formulation
This project focuses on training **soft-margin SVMs** using projection-free and projection-based optimization techniques.  
The dual formulation of the SVM problem is optimized using:
- **Frank–Wolfe algorithm**
- **Projected Gradient Descent**

---

## Methodology

### 1. Frank–Wolfe Algorithm
- **Projection-free** convex optimization.
- Step-size strategies:
  - **Diminishing step size**: γ = 2 / (k+2)
  - **Armijo line search**: adaptive, dynamic selection.
- Convergence monitored using the **duality gap**.

### 2. Projected Gradient Descent (PGD)
- Gradient descent step followed by projection onto the feasible set.
- **Armijo line search** for step-size determination.
- Convergence monitored using **gradient mapping values**.

---

## Datasets

### a4a Dataset
- Adapted from the **Adult dataset**.
- Binary features after quantile transformation and one-hot encoding.
- Train: 1,605 samples; Test: 30,956 samples.
- 123 features.

### Breast Cancer Dataset
- From **LIBSVM repository** (Wisconsin Breast Cancer dataset).
- 10 normalized numerical features.
- Train: 341 samples; Test: 342 samples.

---

## Experimental Setup
- **Iterations**:
  - a4a: 5,000
  - Breast Cancer: 1,000 (FW extended to 3,000 in tests)
- **Stopping tolerance**: 1e-6
- Metrics:
  - CPU time
  - Objective function value
  - Duality gap (FW) / Gradient mapping (PGD)
  - Accuracy (supplementary, due to class imbalance)

---

## Results Summary

### Frank–Wolfe (FW)
- **Armijo line search** → lower objective value & smaller duality gap but higher CPU cost.
- **Diminishing step size** → faster but less precise convergence.
- Larger datasets (a4a) amplify CPU time differences.

### Projected Gradient Descent (PGD)
- More computationally expensive due to projection operations.
- Performed well on smaller datasets (Breast Cancer) with slightly lower loss.
- Struggled to meet stopping criteria for large datasets within iteration limit.

---

### Accuracy (for reference only)

| Model                 | Dataset        | Accuracy (%) |
|----------------------|---------------|--------------|
| FW (Armijo)          | Breast Cancer | 95.32        |
| FW (Diminishing)     | Breast Cancer | 94.74        |
| PGD (Armijo)         | Breast Cancer | 94.74        |
| FW (Armijo)          | a4a           | 85.45        |
| FW (Diminishing)     | a4a           | 82.11        |
| PGD (Armijo)         | a4a           | 83.36        |

---

## Conclusions
- **FW is more efficient** for large datasets due to its projection-free nature.
- **PGD can achieve competitive or slightly better optimization on smaller datasets**, but at a higher computational cost.
- Step-size strategy choice impacts the **speed–accuracy trade-off**.

---

## References
1. Bomze, I.M., Rinaldi, F., Zeffiro, D. (2020). *Frank–Wolfe and Friends: A Journey into Projection-Free First-Order Optimization Methods*. Optimization Methods and Software, 35(5), 1117–1154.
2. Condat, L. (2015). *Fast Projection onto the Simplex and the L1-ball*. Mathematical Programming, 158(1-2), 575–585.
3. Jaggi, M. (2013). *An Equivalence between the Lasso and Support Vector Machines*. arXiv:1301.6978.
4. Jaggi, M. (2013). *Revisiting Frank–Wolfe: Projection-Free Sparse Convex Optimization*. ICML-13.
5. Lacoste-Julien, S., Jaggi, M. (2015). *On the Global Linear Convergence of Frank-Wolfe Optimization Variants*. arXiv:1505.05809.

---

## Dataset Access
- **a4a dataset**: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
- **Breast Cancer dataset**: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

---

## Reproducibility
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/svm-optimization.git
   cd svm-optimization
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download datasets** (place in `data/` directory):
   - a4a
   - Breast Cancer
4. **Run experiments**:
   ```bash
   python main.py --dataset a4a --method fw --step_size armijo
   python main.py --dataset breast_cancer --method pgd --step_size armijo
   ```
5. **View results** in `results/` folder.
