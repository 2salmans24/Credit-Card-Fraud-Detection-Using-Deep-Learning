# Credit-Card-Fraud-Detection-Using-Deep-Learning
Using deep learning to detect credit card fraud.  

## Overview
This project focuses on detecting fraudulent credit card transactions using deep learning and machine learning techniques. Fraud detection is a critical problem in the financial industry, where accurately identifying fraudulent transactions can save significant costs and enhance security.

## Objectives
- Detect fraudulent credit card transactions in an imbalanced dataset.
- Compare deep learning and traditional machine learning models in terms of performance.
- Use techniques like class weights and SMOTE to address the class imbalance.

## Dataset
- **Source**: The dataset is a subset of the Kaggle Credit Card Fraud Detection dataset.
- **Features**:
  - `Time`: Time elapsed since the first transaction.
  - `Amount`: The transaction amount.
  - `V1` to `V28`: Anonymized features resulting from PCA transformation.
  - `Class`: Target variable (0 = Legitimate, 1 = Fraudulent).

## Project Steps
1. **Exploratory Data Analysis (EDA)**:
   - Visualized the class imbalance.
   - Examined distributions of key features (`Amount`, `Time`, and PCA-transformed features).
   - Generated a correlation heatmap for anonymized features (`V1-V28`).

2. **Handling Class Imbalance**:
   - Applied **class weights** during model training.
   - Experimented with **SMOTE** (Synthetic Minority Oversampling Technique) to generate synthetic samples for the minority class.

3. **Model Development**:
   - Built and trained a deep learning model using Keras.
   - Compared the neural network to traditional machine learning models, including Logistic Regression, Random Forest, and Support Vector Machine (SVM).

4. **Evaluation Metrics**:
   - Accuracy
   - AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
   - Confusion Matrix

## Results
| Model                | Accuracy | AUC-ROC |
|----------------------|----------|---------|
| Logistic Regression  | 97.25%   | 0.9822  |
| Random Forest        | 99.97%   | 0.9447  |
| Support Vector Machine (SVM) | 99.78% | 0.9734 |
| Neural Network       | 98.18%   | 0.9708  |

### Key Findings
- **Random Forest** achieved the highest accuracy but had a slightly lower AUC-ROC score.
- **Support Vector Machine (SVM)** and **Neural Network** provided a strong balance between accuracy and AUC-ROC.
- **Logistic Regression** was effective despite being simpler than the other models.

## Files in the Repository
- **`Credit Card Fraud Detection.ipynb`**: Jupyter notebook containing all project steps, including data analysis, preprocessing, model building, and evaluation.
- **`README.md`**: Project documentation.
- **`data/`**: Contains the dataset used for training and testing.

## Requirements
The following Python libraries are required to run the project:
- `tensorflow`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/credit-card-fraud-detection.git

