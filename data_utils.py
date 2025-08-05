# This file contains functions for loading and preparing the datasets.

import numpy as np
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler

def load_a4a_dataset(file_path="a4a"):
    """
    Downloads (if necessary) and loads the a4a dataset in sparse format.
    """
    # Check if the dataset already exists, if not, download it
    if not os.path.exists(file_path):
        print("Downloading a4a dataset...")
        url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a4a"
        urllib.request.urlretrieve(url, file_path)
        print("a4a dataset downloaded successfully.")

    # Load the dataset from the file in sparse format
    print(f"Loading the {file_path} dataset...")
    X, y = load_svmlight_file(file_path)

    # Adjust labels: 1 -> 1, others -> -1
    y = np.where(y == 1, 1, -1)
    
    return X, y

def load_and_prepare_breast_cancer_data(url="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/breast-cancer_scale"):
    """
    Downloads and prepares the breast cancer dataset.
    """
    # Step 1: Download the file
    file_path = "breast-cancer_scale.txt"
    urllib.request.urlretrieve(url, file_path)

    # Step 2: Load the data using load_svmlight_file
    X, y = load_svmlight_file(file_path)
    X = X.toarray()  # Convert sparse matrix to dense format for this dataset

    # Step 3: Transform labels (2 -> 1, 4 -> -1)
    y[y == 2] = 1
    y[y == 4] = -1

    return X, y

def check_and_normalize(X_train, X_test):
    """
    Checks for missing values and normalizes data to the range [-1, 1].
    """
    # Step 1: Check for missing values
    missing_train = np.isnan(X_train).sum()
    missing_test = np.isnan(X_test).sum()

    print(f"Missing values in training set: {missing_train}")
    print(f"Missing values in test set: {missing_test}")

    # Step 2: Normalize data to range [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    print("Data has been normalized to range [-1, 1]")

    return X_train_normalized, X_test_normalized

def check_class_balance(y, dataset_name=""):
    """
    Checks and prints the class balance of a target variable.
    """
    # Count occurrences of each class
    unique_classes, class_counts = np.unique(y, return_counts=True)

    # Calculate total samples and class ratios
    total_samples = len(y)
    class_ratios = {cls: count / total_samples for cls, count in zip(unique_classes, class_counts)}

    # Print class information
    print(f"\n--- Class Distribution for {dataset_name} ---")
    for cls, count in zip(unique_classes, class_counts):
        print(f"Class {int(cls)}: {count} samples ({class_ratios[cls]:.2%} of total)")

    # Check if dataset is imbalanced
    is_imbalanced = any(ratio < 0.4 for ratio in class_ratios.values())
    if is_imbalanced:
        print("The dataset is imbalanced.")
    else:
        print("The dataset is balanced.")
    print("--------------------------------------")
    
    return {"class_counts": dict(zip(unique_classes, class_counts)), "class_ratios": class_ratios}