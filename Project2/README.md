# Breast Cancer Detection Using Logistic Regression

A machine learning project that uses Logistic Regression to classify breast cancer tumors as malignant or benign based on the Wisconsin Breast Cancer Dataset.

## Overview

This project demonstrates a complete machine learning pipeline for binary classification, achieving **98.25% accuracy** in detecting breast cancer using sklearn's built-in breast cancer dataset.

## Features

- **Dataset**: Wisconsin Breast Cancer Dataset (569 samples, 30 features)
- **Model**: Logistic Regression
- **Accuracy**: 98.25%
- **Visualizations**: Confusion Matrix with heatmap
- **Preprocessing**: Standard Scaling for feature normalization

## Dataset Information

- **Total Samples**: 569
- **Features**: 30 numerical features computed from digitized images of fine needle aspirate (FNA) of breast mass
- **Target Classes**: 
  - Malignant (0)
  - Benign (1)
- **Train/Test Split**: 80/20 (455 training, 114 testing samples)

### Key Features Include:
- Mean radius, texture, perimeter, area
- Smoothness, compactness, concavity
- Concave points, symmetry, fractal dimension
- Plus error and "worst" measurements for each

## Requirements

```python
numpy
pandas
matplotlib
scikit-learn
```

## Installation

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

1. Clone the repository or download the notebook
2. Install required dependencies
3. Run the Jupyter notebook:

```bash
jupyter notebook "Breast Cancer Detection.ipynb"
```

## Project Pipeline

### 1. Data Loading
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

### 2. Data Exploration
- Feature shape: (569, 30)
- Target shape: (569,)
- No missing values

### 3. Train-Test Split
- 80% training data (455 samples)
- 20% testing data (114 samples)
- Stratified split to maintain class distribution

### 4. Feature Scaling
- StandardScaler applied to normalize features
- Mean = 0, Standard Deviation = 1

### 5. Model Training
- Algorithm: Logistic Regression
- Max iterations: 10,000
- Solver: lbfgs (default)

### 6. Model Evaluation

**Performance Metrics:**

| Metric | Malignant | Benign |
|--------|-----------|--------|
| Precision | 0.98 | 0.99 |
| Recall | 0.98 | 0.99 |
| F1-Score | 0.98 | 0.99 |

**Confusion Matrix:**
```
[[41  1]
 [ 1 71]]
```

- True Positives (Benign): 71
- True Negatives (Malignant): 41
- False Positives: 1
- False Negatives: 1

## Results

The model achieves:
- **Overall Accuracy**: 98.25%
- **Macro Average F1-Score**: 0.98
- **Weighted Average F1-Score**: 0.98

Only 2 misclassifications out of 114 test samples, demonstrating excellent performance for medical diagnosis support.

## Visualization

The project includes a confusion matrix heatmap that visually represents the model's predictions vs actual labels, making it easy to identify classification patterns and errors.

## How Logistic Regression Works

The model:
1. Finds optimal weights for each of the 30 features
2. Computes probability using the sigmoid function
3. Classifies based on threshold (default: 0.5)
   - Probability > 0.5 → Benign (1)
   - Probability ≤ 0.5 → Malignant (0)

## Future Improvements

- [ ] Hyperparameter tuning (C parameter, solver selection)
- [ ] Feature importance analysis
- [ ] Try other algorithms (Random Forest, SVM, Neural Networks)
- [ ] Cross-validation for more robust evaluation
- [ ] ROC curve and AUC analysis
- [ ] Implement ensemble methods

## Medical Disclaimer

⚠️ **Important**: This is an educational project for demonstration purposes only. It should **NOT** be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## License

This project uses the Wisconsin Breast Cancer Dataset, which is freely available through scikit-learn.

## Author

Created as a machine learning classification demonstration project.

## Acknowledgments

- Dataset: UCI Machine Learning Repository
- Library: scikit-learn development team
- Inspiration: Medical AI applications for early cancer detection

---

**Note**: The high accuracy achieved (98.25%) demonstrates that machine learning can be a powerful tool in assisting medical professionals with diagnosis, though it should always be used as a supplementary tool alongside expert medical judgment.
