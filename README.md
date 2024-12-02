
# Feature Selection Optimization Algorithms

This repository contains implementations of several optimization algorithms for feature selection in machine learning tasks. The algorithms aim to optimize feature subsets that best contribute to predictive performance using classification models, specifically the K-Nearest Neighbors (KNN) classifier.

## Algorithms Included

- **Salp Swarm Optimization (SSO)**
- **Grey Wolf Optimizer (GWO)**
- **Grasshopper Optimization Algorithm (GOA)**
- **Whale Optimization Algorithm (WOA)**

Each algorithm applies optimization strategies to select the most relevant features from a dataset, aiming to maximize classification accuracy while minimizing the number of features.

## Datasets

The following datasets are used for training and evaluation:

- **Hill Valley Dataset**: Used for evaluating classification performance based on selected features.
- **Semeion Handwritten Digit Dataset**
- **Arrhythmia Dataset**

## Installation

You need to have Python and the following libraries installed to run the code:

- numpy
- pandas
- scikit-learn
- datetime

### Requirements

You can install the required dependencies using `pip`. You can create a `requirements.txt` file with the following content:

```txt
numpy
pandas
scikit-learn
```

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

After setting up the environment, you can run the script to test the optimization algorithms on the datasets.

### Example for running the Salp Swarm Optimization (SSO) algorithm:

1. First, modify the `data` variable in the code to point to the desired dataset path (e.g., `hill-valley_csv.csv`, `arrhythmia_csv.csv`, etc.).
2. Then, run the Python script with the algorithm (e.g., Salp, Grey Wolf, Grasshopper, or Whale).

```bash
python your_algorithm_script.py
```

### Results

The script will output the classification accuracy and the number of features selected for each optimization algorithm.

```txt
percentage: 50.0 %
Accuracy: 0.92
Fitness Value: 0.75
...
```

## Algorithms' Details

### 1. **Salp Swarm Optimization (SSO)**

The Salp Swarm Optimization (SSO) algorithm is inspired by the foraging behavior of salps. This algorithm attempts to optimize the feature selection process by simulating the movement and search process of a swarm of salps.

- **Objective**: Select the best subset of features to minimize the error rate and feature count.
  
### 2. **Grey Wolf Optimizer (GWO)**

Grey Wolf Optimizer mimics the leadership hierarchy and hunting behavior of grey wolves. It balances exploration and exploitation to find the optimal solution for feature selection.

- **Objective**: Similar to SSO, the goal is to select features that maximize classification performance using KNN.

### 3. **Grasshopper Optimization Algorithm (GOA)**

The Grasshopper Optimization Algorithm is inspired by the movement and interaction behaviors of grasshoppers. This algorithm is particularly suited for solving optimization problems.

- **Objective**: Similar to other algorithms, it aims to find the most relevant features for classification tasks.

### 4. **Whale Optimization Algorithm (WOA)**

The Whale Optimization Algorithm is inspired by the hunting behavior of humpback whales. It simulates the search for prey and optimizes the feature selection process.

- **Objective**: Identify the subset of features that lead to the best classification accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Inspired by optimization algorithms for feature selection.
- The datasets used are publicly available from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
