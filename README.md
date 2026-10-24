# Predicting Risk Scores for Coronary Artery Disease (CAD)
A machine learning project which outputs the probability of a patient developing Coronary Artery Disease (CAD) from patient measurements.

## ⚠️ Disclaimer
This machine learning project is intended solely for educational and portfolio purposes to demonstrate programming and data science skills. It is not a medical device, has not been clinically validated, and must not be used for diagnosis, prognosis, treatment decisions, or any clinical application.

## 🎯 Project Overview
To develop a supervised machine learning model to predict the probability risk of a patient developing Coronary Artery Disease (CAD), using structured health data.

## 📂 Repository Structure

```
heart_disease_prediction_model/
├── 1_notebooks/
│ ├── 1_data_understanding_cleaning.ipynb
│ └── 2_EDA.ipynb
│ └── 3_Preprocessing.ipynb
│ └── 4_Model_Evaluation.ipynb
│ └── 05_Final_Model.ipynb
├── 2_streamlit_app/
│ └── CAD_project_demo.mp4
│ └── app.py
├── 3_report/
└── README.md
```
## ℹ️ Dataset Information

### Citation

**Name:** Heart Disease

**Source:** UCI Machine Learning Repository

**Citation:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository.

**DOI:** 10.24432/C52P4X

**Link:** https://archive.ics.uci.edu/dataset/45/heart+disease

**License:** CC BY 4.0

### Dataset Structure

**Number of Records:** 303

**Number of Features:** 14

**Target Variable:** Diagnosis of heart disease ('target')

**Data Dictionary:**

|Name|Description|Encoding|Role|
|----|-----------|--------|-----|
|age|age of patient in years|N/A|Feature|
|sex|sex of patient (M/F)|1.0 = Male; 0.0 = Female|Feature|
|cp|type of chest pain|1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic|Feature|
|trestbps|resting blood pressure when admitted to hospital (mmHg)|N/A|Feature|
|chol| serum cholesterol (mg/dl)|N/A|Feature|
|fbs|fasting blood sugar > 120 mg/dl|1 = True; 0 = False|Feature|
|restecg|results from resting electrocardiograph| 0 = normal; 1 = ST-T wave abnormality (T wave inversions/ST elevation or depression > 0.05 mV); 2 = possible/definite left ventricular hypertrophy|Feature|
|thalach|maximum heart rate achieved during exercise (bpm)|N/A|Feature|
|exang|exercise induced angina|1 = yes; 0 = no|Feature|
|oldpeak|ST depression caused by exercise relative to rest|N/A|Feature|
|slope|slope of peak exercise ST segment|1 = upsloping; 2 = flat; 3 = downsloping|Feature|
|ca|number of major blood vessels (LAD, LCx, RCA) coloured by fluoroscopy| 0, 1, 2, 3|Feature|
|thal|thalium stress test result|3 = normal; 6 = fixed defect; 7 = reversible defect|Feature|
|**target**|**diagnosis of heart disease (angiographic disease status)**| **0 - 4 (0 = no diagnosis; 1-4 = diagnosis with '4' being most severe)**|**Target**|

Find the data understanding notebook [here](./1_notebooks/1_data_Understanding.ipynb).

## 📈 Exploratory Data Analysis

**Key Findings:**

* The dataset shows a class imbalance in the target variable, with more individuals not diagnosed with CAD. This imbalance may influence model performance and will require attention during modelling (e.g., resampling or class‑weighted algorithms).
* Among numerical features, oldpeak (ST depression) and thalach (maximum heart rate achieved) show the strongest associations with CAD. Oldpeak is positively correlated with CAD, while thalach is negatively correlated, both aligning with clinical expectations.
* Age and resting blood pressure show weaker but statistically significant relationships with CAD. Cholesterol shows no meaningful univariate association.
* Seven categorical variables show statistically significant differences between CAD and non‑CAD groups: sex, chest pain type (cp), resting ECG (restecg), exercise‑induced angina (exang), slope, number of major vessels (ca), and thal.
* Fasting blood sugar (fbs) is the only categorical variable with no statistical difference between groups (p = 1.00).
* CAD patients are more likely to present with asymptomatic chest pain, flat ST slopes, exercise‑induced angina, abnormal ECG findings, and multiple affected vessels.
* Overall, the dataset contains several clinically meaningful predictors of CAD, providing a strong foundation for feature engineering and model development.

**Visualisations:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/5912da43-8d09-4508-a9e4-d38736477c6e" width="500" />

<p align="center">
  <img src="https://github.com/user-attachments/assets/7006e6a4-f19d-4454-9065-7a8f220b27a8" width="500" />

## ⚙️ Feature Engineering & Preprocessing

**Method:**
* Converted numeric categorical values into readable labels (e.g., 0 -> False, 1 -> True) to improve the interpretability of both the data and model outputs.
* Engineered an `age_bucket` feature, grouping patients into categories such as `young`, `middle-aged`, and `senior`.
* Engineered an `oldpeak_threshold` feature, categorising ST depression values into `low`, `medium`, `high`, and `very high` risk groups.
* Simplified the target variable by mapping 0 -> 0 (no CAD) and 1, 2, 3, 4 -> 1 (presence of CAD), converting the original multi-class diagnosis into a binary classification problem.
* Developed a preprocessing pipeline to prepare features for modelling:
  * Applied One-Hot Encoding to categorical features.
  * Applied StandardScaler to numerical features to standardise feature distribtions.

## ⚖️ Model Evaluation and Selection
* Evaluated three machine learning algorithms for CAD prediction:
  * Logistic Regression
  * Random Forest
  * XGBoost
* Developed and assessed three versions of each model:
  * **Baseline Model:** using default parameters and standard classification threshold.
  * **Threshold-Adjusted Model:** Where the decision threshold was optimised to achieve a recall of 1.00, ensuring all CAD-positive cases were identified.
  * **Hyperparameter-Tuned Model:** Optimised using parameter tuning techniques to improve predictive performance.
* Evaluated each model version using:
  * Classification report (precision, recall, F1-score, and accuracy)
  * Confusion Matrix
  * Comparative performance analysis across evaluation metrics
* Selected the best-performing version of each algorithm and performed a final comparison to identify the most suitable model for CAD risk prediction:
  
  <img width="400" height="106" alt="image" src="https://github.com/user-attachments/assets/919294ae-5a2d-4c98-bd28-453fcbe6ddb1" />

* Prioritised recall as the primary evaluation metric, as minimising false negatives is critical in a clinical screening setting.
* The final selected model was the **Threshold-Adjusted Logistic Regression** model.
* Threshold-Adjusted Logistic Regression outperformed the corresponding Random Forest and XGBoost models by achieving the highest precision, F1 score, and overall accuracy while maintaining a good recall.

## 🌐 Streamlit App
* Deployed the final CAD prediction model as an interaction Streamlit web application.
* Enabled users to enter clinical data through an intuitive user interface.
* Generated a personalised CAD risk score and classification based on the submitted clinical measurements.
* Added a LIME Explanation to provide instance-level explanations for each prediction, showing which ptient features increased or decreased the predicted CAD risk score.
* Incorporated a customisable decision threshold, allowing users to compare predictions made using the default threshold against their own selected threshold.
  * This demonstrated the impact of threshold selection on clinical screening outcomes and the trade-off between recall and precision.
* Displayed key model performance metrics to provide transparency into model performance.
* Included feature importance visualisations to highlight the clinical variables that contribute most strongly to CAD risk predictions.

**[Download and Watch the CAD Risk Prediction Demo Here](2_streamlit_app/CAD_project_demo.mp4)**
