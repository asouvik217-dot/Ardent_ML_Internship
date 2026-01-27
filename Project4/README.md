# Random Forest Classification on Breast Cancer Dataset

## 📌 Project Overview

This project demonstrates how to build, train, and evaluate a **Random Forest classifier** using the **Breast Cancer Wisconsin dataset** from `scikit-learn`. The goal is to predict whether a tumor is **malignant** or **benign** based on computed features from digitized images of breast tissue.

The notebook walks through the full machine learning workflow:

* Data loading and inspection
* Train–test splitting
* Model training using Random Forest
* Model evaluation
* Hyperparameter tuning with GridSearchCV

---

## 📊 Dataset

* **Source:** `sklearn.datasets.load_breast_cancer`
* **Samples:** 569
* **Features:** 30 numerical features (e.g., mean radius, texture, smoothness)
* **Target classes:**

  * `0` – Malignant
  * `1` – Benign

The dataset is clean and does not require missing-value handling.

---

## 🧠 Model Used

**Random Forest Classifier**

* Ensemble method based on multiple decision trees
* Reduces overfitting compared to a single decision tree
* Provides strong performance on tabular data

Key parameters explored:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`

---

## ⚙️ Workflow Steps

1. **Import Libraries** – NumPy, Pandas, scikit-learn utilities
2. **Load Dataset** – Features (`X`) and labels (`y`)
3. **Train-Test Split** – Stratified split to preserve class balance
4. **Train Model** – Random Forest with default parameters
5. **Make Predictions** – On unseen test data
6. **Evaluate Model** – Accuracy, confusion matrix, classification report
7. **Hyperparameter Tuning** – GridSearchCV with cross-validation
8. **Final Evaluation** – Performance using the best model

---

## 📈 Evaluation Metrics

The model is evaluated using:

* **Accuracy**
* **Confusion Matrix**
* **Precision, Recall, F1-score** (via classification report)

These metrics provide insight into both overall performance and class-specific behavior.

---

## 🛠 Requirements

Install the required dependencies before running the notebook:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

Python version: **3.8+** recommended

---

## ▶️ How to Run

1. Clone or download this repository
2. Open the notebook:

   ```bash
   jupyter notebook Project_4_Random_Forest_breast_Cancer_Dataset.ipynb
   ```
3. Run the cells sequentially

---

## ✅ Results

* Random Forest achieves **high accuracy** on the test set
* Hyperparameter tuning further improves generalization
* The model performs well on both malignant and benign classes

---

## 📌 Key Takeaways

* Random Forest is highly effective for structured medical datasets
* Stratified splitting is important for imbalanced or sensitive data
* Hyperparameter tuning can significantly improve performance

---

## 📄 License

This project is for educational purposes and uses publicly available data from `scikit-learn`.

---

## ✍️ Author

Developed as part of a machine learning project on classification using ensemble models.

