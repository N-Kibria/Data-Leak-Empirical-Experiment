# Diabetes Classification with Preprocessing and Sampling Experiments

This project evaluates the effect of different preprocessing pipelines and class-balancing strategies on the classification of diabetes using the PIMA Indians Diabetes Dataset. Multiple machine learning models are trained and compared under each experimental configuration using 5-fold stratified cross-validation.

---

## Dataset

**Name:** PIMA Indians Diabetes Database  
**Source:** Kaggle (`pima-indians-diabetes-database/diabetes.csv`)  
**Size:** 768 samples, 8 features, 1 binary target (`Outcome`)  
**Class distribution:** Imbalanced (approximately 65% negative, 35% positive)

Several features use the value `0` to represent missing data, including `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`.

---

## Project Structure

```
.
├── diabetes_experiment.py    # Main experiment script
└── README.md
```

---

## Preprocessing Steps

Each preprocessing step is configurable with a `mode` parameter:

- `train_only` — the operation is applied only to training data; test data is left unchanged.
- `all` — the operation is applied to both training and test data.

### 1. Missing Value Imputation

Zeros in physiologically implausible columns are treated as missing values and replaced using class-specific median imputation. That is, the median is computed separately for diabetic and non-diabetic samples in the training set, then used to fill missing values in the respective class.

Note: When `mode='all'`, this strategy constitutes target leakage on the test set, as the test target labels are used to guide imputation. This is intentional and documented as part of the experimental design.

### 2. Outlier Removal

Outliers are identified using the interquartile range (IQR) method with a 1.5x multiplier. Boundaries are always computed from the training set. When `mode='all'`, the same boundaries are used to filter the test set as well.

### 3. Normalization

Features are standardized using `StandardScaler`. When `mode='all'`, the scaler is fit separately on the training set and then applied to the test set using training statistics. When `mode='train_only'`, only the training set is normalized.

> Note: In the `all` mode as currently implemented, `scaler.fit_transform` is called on the test set independently, which does not use training statistics for the test set. This is a known inconsistency in the code and differs from the standard `fit`/`transform` split.

### 4. Class Imbalance Sampling

Four sampling strategies are evaluated independently:

| Method | Type | Description |
|---|---|---|
| SMOTE | Oversampling | Generates synthetic minority class samples |
| Tomek Links | Undersampling | Removes majority class samples near class boundaries |
| NearMiss (v1) | Undersampling | Selects majority samples closest to minority samples |
| SMOTE + Tomek | Combined | Applies SMOTE followed by Tomek link removal |

When `sampling_mode='train_only'`, sampling is applied only to the training set. When `sampling_mode='all'`, the sampler is fit and applied independently to both the training and test sets.

---

## Models

The following classifiers are evaluated in each experiment:

- CatBoost
- Random Forest
- XGBoost
- Support Vector Machine (SVM with RBF kernel, probability estimates enabled)
- Extra Trees

All models use `random_state=42` where applicable.

---

## Experiments

The main function `perform_cv_with_preprocessing` accepts configurable parameters for each preprocessing step and runs 5-fold stratified cross-validation. Each sampling method is run as a separate experiment within a single call.

The following configurations were executed:

| Experiment | Normalize | Missing Values | Outlier Removal | Sampling |
|---|---|---|---|---|
| 1 | all | all | all | all (4 methods) |
| 2 | all | all | train_only | all (4 methods) |
| 3 | all | train_only | train_only | all (4 methods) |
| 4 | train_only | train_only | train_only | all (4 methods) |
| 5 | train_only | train_only | train_only | train_only (4 methods) |

---

## Evaluation Metrics

Each model is evaluated on the following metrics per fold, and the mean and standard deviation are reported across all 5 folds:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

---

## Key Results Summary

The table below shows the best average AUC-ROC achieved per sampling method across all configurations.

| Sampling Method | Best AUC-ROC | Configuration |
|---|---|---|
| SMOTE | 0.9513 (RandomForest) | Normalize=all, Missing=all, Outlier=all |
| Tomek Links | 0.9525 (CatBoost) | Normalize=all, Missing=all, Outlier=all |
| NearMiss | 0.9136 (CatBoost) | Normalize=all, Missing=all, Outlier=all |
| SMOTE + Tomek | 0.9656 (ExtraTrees) | Normalize=all, Missing=all, Outlier=all |

Applying preprocessing to both train and test data (including target-leaking imputation and sampling the test set) consistently produced higher reported metrics. Experiments where only training data was preprocessed yielded lower but more realistic estimates of generalization performance.

---

## Requirements

```
scikit-learn==1.2.2
imbalanced-learn==0.10.1
xgboost
catboost
pandas
numpy
```

Install with:

```bash
pip install scikit-learn==1.2.2 imbalanced-learn==0.10.1 xgboost catboost pandas numpy
```

---

## Usage

```python
import pandas as pd
from diabetes_experiment import perform_cv_with_preprocessing

df = pd.read_csv('diabetes.csv')

# Run with all preprocessing applied to both train and test
results = perform_cv_with_preprocessing(
    df,
    normalize_mode='train_only',
    missing_mode='train_only',
    outlier_mode='train_only',
    sampling_mode='train_only'
)
```

---

## Important Caveats

**Target leakage:** The missing value imputation function uses the test set target labels when `mode='all'`. This inflates performance metrics and does not reflect real-world deployment conditions.

**Test set sampling:** Applying SMOTE or other resamplers to the test set changes its class distribution and sample count, which makes the reported metrics on the resampled test set incomparable to a natural held-out evaluation.

**Recommended configuration for honest evaluation:** Set `missing_mode='train_only'`, `outlier_mode='train_only'`, `normalize_mode='train_only'`, and `sampling_mode='train_only'` to ensure no information from the test set influences any preprocessing or training step.

---

## License

This project is provided for educational and research purposes. The dataset is publicly available on Kaggle under its respective license.
