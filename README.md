
# Titanic Survival Prediction: A Machine Learning Analysis

This project explores various machine learning techniques to predict the survival of passengers on the Titanic. We analyze the famous Titanic dataset from Kaggle using five distinct classifiers: Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Bayes Classifier, Decision Trees, and Multi-Layer Perceptron (MLP). Each model is rigorously evaluated and compared to determine the most effective approach.

## Table of Contents

- [Abstract](#abstract)
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Classification Models](#classification-models)
  - [Support Vector Machines (SVM)](#support-vector-machines-svm)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Bayes Classifier](#bayes-classifier)
  - [Decision Trees](#decision-trees)
  - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

The Titanic dataset from Kaggle provides information about passengers aboard the Titanic. The objective is to predict the survival of passengers based on various features such as age, gender, and ticket class. This project explores five classification algorithms, focusing on model performance and generalization ability.

## Data Preprocessing

- **Feature Selection:** Features relevant to survival prediction are selected based on correlation analysis.
- **Data Splitting:** The dataset is split into training, validation, and testing sets.
- **Data Transformation:** Missing values are imputed, categorical features are encoded, and data is standardized using a preprocessing pipeline.

## Classification Models

### Support Vector Machines (SVM)

- **Code Location:** `src/svm_classifier.py`
- **Explanation:** An SVM model with hyperparameter tuning to optimize the classification performance.

### K-Nearest Neighbors (KNN)

- **Code Location:** `src/knn_classifier.py`
- **Explanation:** A KNN model using Euclidean distance for classification, with an optimal value of `k` determined through cross-validation.

### Bayes Classifier

- **Code Location:** `src/bayes_classifier.py`
- **Explanation:** A Naive Bayes classifier assuming feature independence, providing a simple yet effective approach to classification.

### Decision Trees

- **Code Location:** `src/decision_tree.py`
- **Explanation:** A decision tree classifier that splits the dataset based on information gain, with a focus on avoiding overfitting through pruning techniques.

### Multi-Layer Perceptron (MLP)

- **Code Location:** `src/mlp_classifier.py`
- **Explanation:** A neural network model with a single hidden layer, trained using backpropagation.

## Model Evaluation

The models are evaluated using:
- **Accuracy:** The proportion of correctly classified instances.
- **Precision, Recall, and F1-Score:** Metrics to assess the balance between true positive and false positive rates.
- **ROC/AUC Curves:** To visualize and compare the performance of the classifiers.

## Conclusion

- **Model Comparison:** The MLP and Decision Tree models showed the highest accuracy on the test set, while the SVM and Bayes classifiers provided consistent results with moderate computational efficiency.
- **Efficiency Considerations:** The MLP model, while accurate, required the most computational resources, making it less efficient for larger datasets.

## Installation

To run this project, you need Python 3.x and the required packages.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
