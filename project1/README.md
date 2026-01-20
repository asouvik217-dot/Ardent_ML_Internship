# Python Basics and Machine Learning Introduction

This repository contains a Jupyter notebook demonstrating fundamental Python concepts and introductory machine learning techniques.

## Contents

### 1. Python Fundamentals
- **Data Types & Variables**: Working with integers, strings, lists, tuples, sets, and dictionaries
- **String Operations**: String manipulation including reversal using slicing
- **Type Handling**: Understanding type constraints in Python (e.g., string concatenation rules)
- **Control Structures**: Conditional statements and control flow
- **Date/Time Operations**: Using the `datetime` module
- **Mathematical Operations**: Working with the `math` module

### 2. Data Visualization
- **Bar Charts**: Visualizing student performance across subjects using Matplotlib
- **Scatter Plots**: Displaying relationships between variables
- **Customization**: Adding titles, labels, legends, and grid lines to plots

### 3. Machine Learning Examples

#### Linear Regression
- **Use Case**: Predicting student marks based on study hours
- **Implementation**: Using scikit-learn's `LinearRegression`
- **Visualization**: Plotting actual data points vs. fitted regression line
- **Key Concept**: Understanding the relationship between continuous variables

#### Classification (Iris Dataset)
- **Dataset**: Famous Iris flower dataset (3 species)
- **Features**: Using sepal length and sepal width for 2D visualization
- **Algorithm**: K-Nearest Neighbors (KNN) classifier
- **Visualization**: Scatter plot showing different flower species
- **Model Performance**: Achieved 80% test accuracy

## Key Libraries Used

```python
import datetime
import math
import pandas
import matplotlib.pyplot as plt
import tensorflow
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
```

## Notable Examples

### Linear Regression Model
The notebook demonstrates a simple linear regression model predicting student marks based on study hours:
- **Input**: Hours studied (1-8 hours)
- **Output**: Predicted marks
- **Model**: `marks ≈ 7.08 × hours + 27.5`

### Iris Classification
- **Dataset Split**: 80% training, 20% testing
- **Features**: 2D visualization using first two features
- **Accuracy**: 80% on test set with K=5 neighbors

## Learning Outcomes

This notebook is ideal for:
- Understanding Python syntax and basic operations
- Learning data visualization with Matplotlib
- Introduction to supervised learning (regression and classification)
- Practical implementation of scikit-learn models
- Understanding the machine learning workflow (data → model → evaluation)

## Running the Notebook

This notebook was created in Google Colab and includes all necessary imports. To run:
1. Open in Google Colab or Jupyter Notebook
2. Run cells sequentially
3. Observe outputs and visualizations

## Notes

- The regression example uses a small synthetic dataset (8 data points)
- The Iris classification uses only 2 features for easy visualization
- All visualizations include proper labels, titles, and legends for clarity
