# Credit Card Fraud Detection Using Machine Learning

A machine learning project for detecting fraudulent credit card transactions using **data balancing, statistical sampling techniques, feature scaling, and multiple classification algorithms**.

The project investigates how different sampling strategies affect the performance of machine learning models on a credit card fraud detection dataset.

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem in which fraudulent transactions represent a small proportion of total transactions.

This project addresses the class-imbalance problem using **SMOTE (Synthetic Minority Over-sampling Technique)** and compares the performance of five machine learning models across multiple sampling strategies.

### Objectives

- Load and preprocess credit card transaction data.
- Separate features and the target variable.
- Address class imbalance using SMOTE.
- Calculate statistical sample sizes.
- Generate samples using different sampling techniques.
- Train multiple machine learning classification models.
- Standardize features before model training.
- Compare model accuracy across different samples.
- Identify the sampling/model combination with the strongest accuracy.

## Workflow

```text
Credit Card Transaction Dataset
            │
            ▼
       Data Loading
            │
            ▼
    Feature / Target Split
            │
            ▼
      SMOTE Balancing
            │
            ▼
     Statistical Sampling
            │
     ┌──────┼─────────┐
     ▼      ▼         ▼
 Random  Systematic  Stratified
     │      │         │
     └──────┼─────────┘
            ▼
      Cluster Sampling
            │
            ▼
    Random Sampling #2
            │
            ▼
     Feature Scaling
            │
            ▼
     Train/Test Split
            │
            ▼
     Machine Learning
            │
     ┌──────┼──────────────┐
     ▼      ▼      ▼       ▼       ▼
    LR     KNN     DT     SVM      RF
            │
            ▼
      Accuracy Comparison
```

## Dataset

The notebook loads the dataset from:

```text
creditcard.csv
```

The feature matrix consists of the transaction attributes, while the final column is treated as the target variable.

The displayed data contains:

- `Time`
- `V1` – `V28`
- `Amount`
- `Target`

The notebook contains **31 columns** after the target is added to the sampled dataset.

> **Note:** The raw `creditcard.csv` dataset is not included in this repository by default.

## Data Balancing

Because fraud detection datasets are typically highly imbalanced, the project applies **SMOTE** before performing the sampling experiments.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
```

This creates synthetic minority-class observations to produce a balanced dataset for the subsequent experiments.

## Sampling Techniques

Five sampling approaches are implemented:

### 1. Simple Random Sampling

Random observations are selected from the balanced dataset.

### 2. Systematic Sampling

Observations are selected at a calculated interval across the dataset.

### 3. Stratified Sampling

Observations are sampled separately according to the `Target` class.

### 4. Cluster Sampling

The balanced dataset is divided into clusters, after which selected clusters are combined into a sample.

### 5. Simple Random Sampling — Different Random State

A second random sample is generated using a different random state to examine variation between random samples.    

## Machine Learning Models

The project compares five classification algorithms:

| Model | Algorithm |
|---|---|
| M1 | Logistic Regression |
| M2 | K-Nearest Neighbors |
| M3 | Decision Tree |
| M4 | Support Vector Classifier |
| M5 | Random Forest |

The models are implemented using `scikit-learn`.

## Feature Scaling

`StandardScaler` is used to standardize the features before model training.

The data is divided into training and testing subsets using an **80/20 split**:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_sample,
    y_sample,
    test_size=0.2,
    random_state=42
)
```

The scaler is fitted on the training data and then applied to both training and testing data.

## Results

The notebook evaluates each model using **classification accuracy**.

| Model | Sampling 1 | Sampling 2 | Sampling 3 | Sampling 4 | Sampling 5 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 96.10% | 96.10% | 96.75% | 97.66% | 94.81% |
| KNN | 96.10% | 96.10% | 95.45% | **98.96%** | 92.21% |
| Decision Tree | 98.70% | 96.10% | 96.10% | 97.92% | 90.91% |
| SVM | 94.81% | 96.10% | 95.45% | 97.92% | 93.51% |
| Random Forest | 98.70% | **98.70%** | **97.40%** | **99.06%** | 93.51% |

The highest recorded accuracy in the notebook is **99.06%**, achieved by the **Random Forest model on Sampling 4**.

## Key Findings

- **Random Forest** produced consistently strong results across the sampling experiments.
- The highest recorded accuracy was **99.06%**.
- **KNN** achieved 98.96% on Sampling 4 but performed substantially worse on Sampling 5.
- Decision Tree achieved 98.70% on Sampling 1.
- Model performance varies considerably depending on the sampling strategy.
- Sampling methodology can therefore materially influence the observed performance of a fraud-classification model.

## Technologies Used

- **Python 3.10**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **imbalanced-learn**
- **Jupyter Notebook**

## Project Structure

```text
credit-card-fraud-detection/
│
├── Credit-Card-Fraud-Detection.ipynb
├── README.md
├── .gitignore
└── creditcard.csv
```

> `creditcard.csv` should generally be excluded from Git if the dataset is large, restricted, or subject to redistribution limitations.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn imbalanced-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Credit-Card-Fraud-Detection.ipynb
```

## Reproducibility

The project uses fixed random states in several experiments, including:

```python
random_state=42
```

This helps make the sampling and train/test experiments reproducible.

## Limitations

This project is primarily an **academic machine learning and sampling experiment** rather than a production-ready fraud detection system.

The current notebook evaluates models using **accuracy only**. For a production-oriented fraud detection system, additional metrics such as:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

would be important, particularly because fraud detection is fundamentally an imbalanced classification problem.

Another important improvement would be applying SMOTE **only to the training data after the train/test split**, rather than balancing the complete dataset before sampling. This avoids potential information leakage between training and testing data.

## Future Improvements

- Add comprehensive EDA and visualization.
- Add confusion matrices.
- Evaluate Precision, Recall, F1-score, ROC-AUC and PR-AUC.
- Use cross-validation.
- Apply SMOTE within the training pipeline.
- Perform hyperparameter tuning.
- Compare additional ensemble models.
- Analyze feature importance.
- Add model explainability using SHAP.
- Build an interactive fraud-prediction application using Streamlit.
- Deploy the best-performing model as an API.

## Author

**Saksham Sahu**

B.Tech — Mathematics & Computing

Interested in **Data Science, Machine Learning, Generative AI, and Agentic AI**.

---

⭐ If you find this project useful, consider giving the repository a star.
