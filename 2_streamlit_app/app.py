import streamlit as st
import joblib, json
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
sns.set_palette(["#5A2D82", "#1A1A1D"])
plt.rcParams["figure.facecolor"] = "#FFFFFF"
plt.rcParams["axes.edgecolor"] = "#E0E0E0"
plt.rcParams["axes.labelcolor"] = "#1A1A1D"
plt.rcParams["xtick.color"] = "#1A1A1D"
plt.rcParams["ytick.color"] = "#1A1A1D"
plt.rcParams["text.color"] = "#1A1A1D"

st.set_page_config(page_title = "CAD Risk Prediction",
                   layout = "wide",
                  initial_sidebar_state = "expanded")

st.markdown("""
<style>

    /* Main background */
    .main {
        background-color: #FFFFFF;
    }

    /* Card-style containers */
    .tech-card {
        background-color: #F5F5F7;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border: 1px solid #E0E0E0;
    }

    /* Headings in darker purple */
    h1, h2, h3, h4 {
        color: #5A2D82 !important;
        font-weight: 600;
    }

    /* Buttons */
    .stButton>button {
        background-color: #5A2D82;
        color: white !important;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        border: none;
        transition: 0.2s;
    }

    .stButton>button:hover {
        background-color: #4A256E;
    }

    /* Slider labels */
    .stSlider label {
        color: #1A1A1D !important;
    }

    /* Metrics */
    .stMetric {
        background-color: #F5F5F7;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        color: #1A1A1D !important;
    }
    
</style>
""", unsafe_allow_html=True)

#Load the pipeline
pipeline_lr = joblib.load("../Final_Model_Components/final_model.pkl")

#Load the threshold
with open("../Final_Model_Components/threshold.json") as f:
    threshold = json.load(f)["threshold"]

#Load feature names
with open("../Final_Model_Components/feature_names.json") as f:
     feature_names = json.load(f)

#Load class names
with open("../Final_Model_Components/class_names.json") as f:
    class_names = json.load(f)

with open("../Final_Model_Components/coefficients.json", "r") as f:
    coef_df = pd.read_json(f)

#Load X_train
X_train = pd.read_json("../Final_Model_Components/X_train.json")

#Load y_train
y_train = pd.read_json("../Final_Model_Components/y_train.json")

#load clean data for eda
df = pd.read_csv('../Data/clean_df.csv')

#Define a prediction function
def predict_single(input_dict):
    #Convert input into one-row df
    df = pd.DataFrame([input_dict])
    #Calculate probability
    proba = pipeline_lr.predict_proba(df)[0,1]
    #Apply threshold
    label = int(proba >= threshold)
    #map to readable label
    actual_label = class_names[label] if 'class_names' in globals() else label
    #Return probability and label
    return proba, actual_label, label

#Calculate probabilities based off training data
y_proba = pipeline_lr.predict_proba(X_train)[:,1]

#Predictions using your threshold
y_pred_thresh = (y_proba >= threshold).astype(int)

#Metrics
#acc = accuracy_score(y_train, y_pred_thresh)
#prec = precision_score(y_train, y_pred_thresh)
#rec = recall_score(y_train, y_pred_thresh)
#f1 = f1_score(y_train, y_pred_thresh)

#Compute curves (ROC + PR)
fpr, tpr, _ = roc_curve(y_train, y_proba)
prec_curve, rec_curve, thresholds = precision_recall_curve(y_train, y_proba)
#cm = confusion_matrix(y_train, y_pred_thresh)

#Create Lime Explainer
explainer = LimeTabularExplainer(
    training_data = pipeline_lr.named_steps['preprocess'].transform(X_train),
    feature_names = feature_names,
    class_names = class_names,
    mode = "classification"
)


#Configure App
#st.set_page_config(page_title = "CAD Risk Prediction",
#                   layout = "wide",
#                  initial_sidebar_state = "expanded")



st.title("CAD Risk Prediction Model")
st.caption("Threshold-Optimised Logistic Regression - Recall-Focused")

tab1, tab2, tab3, tab4 = st.tabs(
    ['Overview', 'Prediction', 'Model Performance', 'Feature Insights']
)

with tab1:
    st.header("Overview")
    with st.container(border = True):
        st.subheader("Purpose")
        st.write("The purpose of this dashboard is to provide an interpretable probability and prediction of Coronary Artery Disease (CAD), using key patient clinical features. It aims to demonstrate how machine learning can aid clinicians and users to assess CAD risk during screening and understand the factors which influence the model's decision.")
    with st.container(border = True):
        st.subheader("How the Model Works")
        st.write("""
        1. Input features collected from patient clinical data.
        2. Preprocessing of the data includes calculating whether data is above thresholds, scaling of numeric data, and encoding of categorical data.
        3. Logistic Regression model predicts the probability of a patient having CAD.
        4. Outputs include risk score (probability), prediction (CAD or no CAD), and Lime Explainability.
        """)
    with st.container(border = True):
        st.subheader("Dataset")
        st.write("""
        The dataset used to train this model contains clinical measurements and biometric measurements which are commonly utilised in CAD risk assessments. Examples include: age, cholesterol, resting blood pressure, max. heart rate, and exercise-induced angina.

This dataset was sourced from UCI Machine Learning Repository.

**Citation:**

*Heart Disease. UCI Machine Learning Repository. DOI: 10.24432/C52P4X.* 

*Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository.*
        """)
    with st.container(border = True):
        st.subheader("Features Used by Model")
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
            - Age
            - Sex
            - Chest pain type
            - Resting blood pressure
            - Cholesterol
            - Fasting blood sugar
            - Thalium Stress Test Results
            """)
        with col2:
            st.write("""
            - Resting ECG
            - Max heart rate
            - Exercise-induced angina
            - ST depression
            - Slope
            - Fluoroscopy Results
            """)

    with st.container(border = True):
        st.subheader("Interpreting the Output")
        st.write("The model outputs a probability between 0 and 1, which indicates the likelihood of CAD.")

    with st.container(border = True):
        st.subheader("Additional Insights")
        st.write("""
        - Model evaluation metrics for default threshold
        - Feature importance visualisations
        """)

    with st.container(border = True):
        st.subheader("Disclaimer")
        st.write("""
        **This tool is purely for education purposes only, with the sole intention of being used to demonstrate ML skills. It is not intended for any kind of clinical use or medical decision-making.** 
        """)
    
        

    

with tab2:
    st.header("Calculate Probability of CAD")
    st.subheader("Enter Patient Information")

    col1, col2 = st.columns([1, 1])
    #Numeric Inputs
    with col1:
        age = st.number_input("Age", min_value = 1, max_value = 120, value = 50)
    
        def compute_age_bucket(age):
            if age < 45:
                return "Young"
            elif age < 60:
                return "Middle-Aged"
            else:
                return "Senior"

        sex = st.selectbox("Sex", options = ["Male", "Female"])

        cp = st.selectbox("Chest Pain (cp)", options = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
        
        trestbps = st.number_input("Resting Blood Pressure (trestbps - mmHg)", min_value = 80, max_value = 200, value = 120)
    
        chol = st.number_input("Cholesterol (chol - mg/dl)", min_value = 100, max_value = 600, value = 150)
    
        thalach = st.number_input("Max Heart Rate (thalach - bpm)", min_value = 60, max_value = 220, value = 150)
    
        oldpeak = st.number_input("ST Depression (oldpeak - mm)", min_value=0.0, max_value = 10.0, value = 1.0, step = 0.1)
    
        def compute_oldpeak_threshold(oldpeak):
            if oldpeak <= 1.0:
                return "Low"
            elif oldpeak <= 2.0:
                return "Medium"
            elif oldpeak <= 5.0:
                return "High"
            else:
                return "Very High"
                
    with col2:    
        fbs = st.selectbox("Fasting Blood Glucose (fbs) > 120 mg/dl", options = ["True", "False"])
        restecg = st.selectbox("Resting ECG Results (restecg)", options = ['Normal', 'ST-T Wave Abnormality', 'Possible Left Ventricle Hypertrophy'])
        exang = st.selectbox("Exercise-induced Angina (exang)", options = ['Yes', 'No'])
        slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", options = ['Flat', 'Upsloping', 'Downsloping'])
        ca = st.selectbox("No. of Major Blood Vessels Coloured in Fluoroscopy (ca)", options = [0, 1, 2, 3])
        thal = st.selectbox("Thalium Stress Test Result (thal)", options = ['Normal', 'Fixed Defect', 'Reversible Defect'])

    input_dict = {
        "age": age,
        "age_bucket": compute_age_bucket(age),
        "trestbps":trestbps,
        "chol":chol,
        "thalach": thalach,
        "oldpeak": oldpeak,
        "oldpeak_threshold": compute_oldpeak_threshold(oldpeak),
        "sex": sex,
        "cp": cp,
        "fbs": fbs,
        "restecg": restecg,
        "exang": exang,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        }
    
    threshold_slider = st.slider(
        "Adjust Classification Threshold",
        min_value = 0.00,
        max_value = 1.00,
        value = threshold,
        step = 0.001,
        format = "%.3f")
    st.caption("""
        The default threshold of 0.133 is chosen to maximise recall for CAD screening. Use the slider to explore how an increased or decreased threshold affects the predictions.
        """)
    
    if st.button("Predict CAD Risk"):
        proba, actual_label, label = predict_single(input_dict)

        #override the label using the slider
        label_slider = int(proba >= threshold_slider)
        actual_label_slider = class_names[label_slider]

        st.subheader("Prediction Result")
        st.metric("CAD Probability", f"{proba:.3f}")
        st.progress(proba)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Threshold-Adjusted Prediction (Default):")
            st.metric("Threshold", threshold)
            st.metric("Prediction", actual_label)

        with col2:
            st.subheader("Slider Threshold Prediction")
            st.metric("Threshold", threshold_slider)
            st.metric("Prediction", actual_label_slider)
        #st.write(f"**Classification:** {actual_label}")
        #st.write(f"Prediction Based on Slider", actual_label_slider)
        #st.write(f"**Threshold Used:** {threshold_slider}")

        #if proba >= threshold + 0.15:
        #    confidence = "High Confidence"
        #elif proba >= threshold:
        #    confidence = "Medium Confidence"
        #else:
        #    confidence = "Low Confidence"
        #st.write(f"**Confidence Level:** {confidence}")

        st.subheader("LIME Explanation")
        user_df = pd.DataFrame([input_dict])
        user_processed = pipeline_lr.named_steps['preprocess'].transform(user_df)
        exp = explainer.explain_instance(
            data_row = user_processed[0],
            predict_fn = pipeline_lr.named_steps["classifier"].predict_proba,
            num_features = 10
        )
        html = exp.as_html()
        # Inject CSS directly into the LIME HTML
        custom_css = """
        <style>
        body {
            font-family: 'Inter', sans-serif !important;
            font-size: 15px !important;
            color: #1A1A1D !important;
        }
        h1, h2, h3 {
            font-family: 'Inter', sans-serif !important;
            color: #5A2D82 !important;
        }
        table {
            font-family: 'Inter', sans-serif !important;
            font-size: 15px !important;
        }
        /* Round the bar corners */
        svg rect {
            rx: 6px !important;
        }
        
        /* Make bar edges smoother */
        svg rect {
            shape-rendering: geometricPrecision !important;
        }
        
        /* Clean up axis text */
        svg text {
            font-family: 'Inter', sans-serif !important;
            font-size: 14px !important;
            fill: #1A1A1D !important;
        }
        
        /* Remove harsh borders */
        svg line {
            stroke: #CCCCCC !important;
        }

        </style>
        """
        html = custom_css + html
        st.components.v1.html(html, height = 600, scrolling = True)


with tab3:

    st.header("Model Performance")
    st.write("Adjust the threshold to see how it affects evaluation metrics.")
    
    if "threshold_slider" not in st.session_state:
        st.session_state["threshold_slider"] = 0.133
        
    if st.button("Return to Default Threshold"):
        st.session_state["threshold_slider"] = 0.133

    threshold = st.slider(
    "Adjust classification threshold",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state["threshold_slider"],
    step=0.001,
    format = "%.3f",
    key = "threshold_slider")

    st.write(f"Current threshold: {threshold:.3f}")
    
    #threshold = st.session_state["threshold_slider"]

    y_pred_thresh = (y_proba >= threshold).astype(int)

    #Recompute metrics
    acc = accuracy_score(y_train, y_pred_thresh)
    prec = precision_score(y_train, y_pred_thresh)
    rec = recall_score(y_train, y_pred_thresh)
    f1 = f1_score(y_train, y_pred_thresh)
    cm = confusion_matrix(y_train, y_pred_thresh)

    st.subheader('Evaluation Metrics')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border = True):
            st.metric("Recall", f'{rec:.3f}')
    with col2:
        with st.container(border = True):
            st.metric("Precision", f'{prec:.3f}')
    with col3:
        with st.container(border = True):
            st.metric("Accuracy", f'{acc:.3f}')
    with col4:
        with st.container(border = True):
            st.metric("F1 Score", f'{f1:.3f}')
    
    #Threshold Summary
    st.subheader("Threshold Summary")
    st.write(f'Current threshold: **{threshold}**')
    st.write("The model is optimised for **recall**, minimising the incidences of false negatives and prioritising the detection of at-risk patients.")

    col_left, col_right = st.columns(2)

    with col_left:
        with st.container(border=True):
            st.subheader("ROC Curve")
            auc = roc_auc_score(y_train, y_proba)
            fig_roc, ax_roc = plt.subplots(figsize = (6,4))
            ax_roc.plot(fpr, tpr, color = '#5A2D82', label = f"AUC = {auc:.3f}")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc = "lower right")
            st.pyplot(fig_roc)

    with col_right:
        with st.container(border=True):
            st.subheader("Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize = (5,3))
            sns.heatmap(cm, annot = True, fmt = "d", cmap = "Purples", ax = ax_cm)
            ax_cm.set_xlabel("Predicted Values")
            ax_cm.set_ylabel("Actual Values")
            st.pyplot(fig_cm)

    #col_one, col_two = st.columns(2)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        #Precision-Recall Curve
        with st.container(border=True):
            st.subheader("Precision-Recall Curve")
            #threshold_to_mark = 0.133
            idx = np.argmin(np.abs(thresholds - threshold))
            prec_point = prec_curve[idx]
            rec_point = rec_curve[idx]
            fig_pr, ax_pr = plt.subplots(figsize = (6,4))
            ax_pr.plot(rec_curve, prec_curve)
            # Threshold marker (lighter purple)
            ax_pr.scatter(
                rec_point, prec_point,
                color="#C8A2C8",        # lighter purple
                s=80,
                edgecolor="#5A2D82",
                linewidth=1.5,
                label=f"Threshold = {threshold: .3f}")
            ax_pr.set_xlabel('Recall')
            ax_pr.set_ylabel('Precision')
            ax_pr.set_title('Precision-Recall Curve')
            st.pyplot(fig_pr)

    #with col2:
     #   with st.container(border=True):
      #      st.subheader("Explanation:")
       #     st.write("")

with tab4:
    st.header("Feature Insights")
    st.write("""
    These insights show how each feature influences the model's prediction.
    Positive coefficents increase CAD risk, while negative coefficients reduce it.
    Odds rations > 1 indicate increased risk.""")
    col_1, col_2 = st.columns(2)
    with col_1:
        #Display table
        with st.container(border=True):
            st.subheader("Coefficient & Odds Ratio Table")
            st.dataframe(coef_df.sort_values("importance", ascending = False)[["feature", "odds_ratio", "importance"]])
    with col_2:
        with st.container(border=True):
            st.subheader("Feature Importance (Absolute Coefficient)")
            fig_imp, ax_imp = plt.subplots(figsize = (6,8))
            coef_df = coef_df.sort_values("importance", ascending=False)
            ax_imp.barh(coef_df["feature"], coef_df["importance"])
            ax_imp.invert_yaxis()
            ax_imp.set_xlabel("Importance (coefficient)")
            st.pyplot(fig_imp)
    with st.container(border=True):    
        st.subheader("Clinical Interpretation")
        top_features = coef_df.sort_values("importance", ascending = False).head(5)
        st.write("**The top drivers of CAD risk in this logistic regression model:**")
        for _,row in top_features.iterrows():
            direction = "increases" if row["importance"] > 0 else "reduces"
            st.write(f"- **{row['feature']}** {direction} predicted CAD risk")