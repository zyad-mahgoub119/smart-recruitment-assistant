# Smart Recruitment Assistant — HR Analytics: Job Change of Data Scientists

Predicting whether a data scientist enrolled in HR-sponsored training is likely to look for a
new job, so recruiters/HR teams can prioritize outreach and retention efforts more efficiently.

## Project Background

Modern organizations receive hundreds of applications per vacancy, and HR teams increasingly
lean on data-driven systems to support hiring and retention decisions. This project builds a
binary classification model that predicts whether a candidate is likely to change jobs (`target
= 1`) or stay (`target = 0`), based on demographic, educational, and employment-history features
collected during training enrollment.

## Dataset

The [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)
dataset (from Kaggle) contains three files:

| File | Rows | Description |
|---|---|---|
| `aug_train.csv` | 19,158 | Labeled training data |
| `aug_test.csv` | 2,129 | Unlabeled data used for final hold-out evaluation |
| `sample_submission.csv` | — | Submission format reference |

**Features** (14 columns, excluding `enrollee_id` and `target`):

`city`, `city_development_index`, `gender`, `relevent_experience`, `enrolled_university`,
`education_level`, `major_discipline`, `experience`, `company_size`, `company_type`,
`last_new_job`, `training_hours`

**Target:** `target` — `1` if the candidate is looking for a job change, `0` otherwise.

> The dataset ships with the repo as a zip archive (`HR_Analytics__Job_Change_of_Data_Scientists_Dataset.zip`).
> Unzip it before running the notebook.

## Approach

The full workflow lives in
[`HR_Analytics_Job_Change_of_Data_Scientists.ipynb`](./HR_Analytics_Job_Change_of_Data_Scientists.ipynb):

1. **Exploratory Data Analysis**
   - Missing-value audit — found heavy missingness in `company_type` (32%), `company_size` (31%),
     `gender` (24%), and `major_discipline` (15%).
   - Missingness was shown to be *informative* rather than random: rows missing `company_size`/
     `company_type` had a job-change rate roughly double that of complete rows.
   - Distribution checks on numeric features (`city_development_index`, `training_hours`) and
     category-frequency / target-rate breakdowns for categorical features.

2. **Preprocessing pipeline** (`scikit-learn` `ColumnTransformer`, fit only on the training split
   to avoid leakage)
   - `training_hours`: log-transform (right-skewed) → median impute → scale
   - `city_development_index`: impute → scale
   - `relevent_experience`: one-hot encoded binary flag
   - High-missingness nominal columns (`gender`, `major_discipline`, `company_type`,
     `enrolled_university`): missing values filled with an explicit `"Missing"` category, then
     one-hot encoded
   - Ordinal columns (`education_level`, `experience`, `company_size`, `last_new_job`): encoded
     with an explicitly defined low-to-high order (not sklearn's default alphabetical order)
   - `city`: custom frequency encoder (replaces each city with its training-set frequency) to
     avoid a huge sparse one-hot matrix

3. **Modeling** — five classifiers trained on the same processed features and compared on a
   stratified 80/20 train/validation split, then on the real `aug_test.csv` labels:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - LightGBM
   - CatBoost

4. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves
   for every model, on both validation and test data.

## Results

| Model | Val Accuracy | Val ROC-AUC | Test Accuracy | Test ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.780 | 0.802 | 0.764 | 0.780 |
| Random Forest | 0.795 | 0.811 | 0.791 | 0.796 |
| XGBoost | 0.790 | 0.819 | 0.788 | 0.799 |
| LightGBM | 0.791 | 0.823 | 0.788 | 0.796 |
| **CatBoost** | 0.785 | 0.821 | 0.784 | **0.801** |

**Key findings:**

- **CatBoost generalized best** on the real test set (ROC-AUC 0.801), closely followed by
  XGBoost (0.799); all five models land in a fairly tight 0.78–0.80 band.
- **Logistic Regression lags badly on recall** (~0.33), missing roughly two-thirds of candidates
  who actually change jobs. Tree-based models catch 67–78% of true job-changers, which matters
  more than raw accuracy for an HR use case focused on not missing at-risk candidates.
- The validation → test ROC-AUC drop was small and consistent (~0.015–0.027) across all models,
  consistent with normal sampling variation rather than overfitting.
- `city_development_index`, `company_size`, and `experience` were consistently the top predictive
  features across tree-based models, aligning with the EDA finding that missing company
  information itself carries signal.

**Recommendation:** CatBoost or XGBoost as the production candidate — both combine the strongest
test-set ROC-AUC with usable recall, and both showed stable validation-to-test performance.

**Limitations / next steps:**
- No hyperparameter tuning was performed (all boosted models used fixed, reasonable defaults).
- The classification threshold was left at 0.5 rather than tuned to the HR team's real cost
  trade-off between false positives (unnecessary retention outreach) and false negatives (an
  unflagged leaver).
- SHAP values would give a more rigorous feature-attribution story than built-in importances.

## Repository Structure

```
.
├── Jupyternotebooks_HR_Analytics_Job_Change_of_Data_Scientists.ipynb   # Full analysis & modeling notebook
├── HR_Analytics__Job_Change_of_Data_Scientists_Dataset.zip             # Raw dataset (train/test/sample submission)
├── Project1.pdf                                                         # Project brief / problem statement
└── README.md
```

## Getting Started

1. Clone the repo and unzip the dataset:
   ```bash
   unzip HR_Analytics__Job_Change_of_Data_Scientists_Dataset.zip -d dataset
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook Jupyternotebooks_HR_Analytics_Job_Change_of_Data_Scientists.ipynb
   ```

## Authors

Youssef Moataz, Mariam Mohamed, Zyad Mahgoub, Abdullah Hany, Reem Ahmed
