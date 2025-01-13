import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the trained model and scaler
model = joblib.load("logistic_regression.joblib")
scaler = joblib.load("scaler2.joblib")

# Title of the web app
st.title("NSCLC Prediction App")
st.write("This app predicts whether a patient has NSCLC or is healthy based on selected features.")

# Define the input features required for the model
selected_features = [
    'Indole-3-propionic acid', 'Indole-3-acrylic acid', 'C5-OH', 'C16-1', 
    'Indole-3-carboxaldehyde', 'ADMA', 'C18', 'C5-DC', 'C6-DC', 'C8-1', 
    'C10-1', 'Quinolinic acid', 'C14-2', 'Cit', 'Trp', 'C18-OH'
]

# Create tabs for input methods
tab1, tab2 = st.tabs(["Manual Input", "Upload Excel File"])

# --- Input Method: Manual Input ---
with tab1:
    st.header("Manual Input")
    st.write("Enter metabolite values manually.")
    user_input = {}
    metabolite_info = {
        'Indole-3-propionic acid': "Indole-3-propionic acid",
        'Indole-3-acrylic acid': "Indole-3-acrylic acid",
        'C5-OH': "Hydroxyvaleryl carnitine",
        'C16-1': "Hexadecenoyl carnitine",
        'Indole-3-carboxaldehyde': "Indole-3-carboxaldehyde",
        'ADMA': "Asymmetric dimethylarginine",
        'C18': "Octadecanoyl carnitine",
        'C5-DC': "Glutaryl carnitine",
        'C6-DC': "Adipyl carnitine",
        'C8-1': "Octenoyl carnitine",
        'C10-1': "Decenoyl carnitine",
        'Quinolinic acid': "Quinolinic acid",
        'C14-2': "Tetradecadienoyl carnitine",
        'Cit': "Citrulline",
        'Trp': "Tryptophan",
        'C18-OH': "Hydroxyoctadecanoyl carnitine"
    }


    for feature in selected_features:
        user_input[feature] = st.number_input(
            f"{feature}:", 
            step=0.00001, 
            format="%.5f", 
            help=metabolite_info[feature]
        )
    manual_input_df = pd.DataFrame([user_input])

# --- Input Method: Upload Excel File ---
with tab2:
    st.header("Upload Excel File")
    with st.expander("📂 Excel File Format Instructions"):
        st.write(
        """
        To upload patient data via an Excel file, ensure the file has the following format:
        - The file must include these column names exactly:
          - **Indole-3-propionic acid**, **Indole-3-acrylic acid**, **C5-OH**, **C16-1**, 
            **Indole-3-carboxaldehyde**, **ADMA**, **C18**, **C5-DC**, **C6-DC**, **C8-1**, 
            **C10-1**, **Quinolinic acid**, **C14-2**, **Cit**, **Trp**, **C18-OH**.
        - Each row should represent a patient's metabolite values.
        """
        )
    uploaded_file = st.file_uploader("Upload Patient Data (Excel)", type=["xlsx", "xls"])
    if uploaded_file:
        uploaded_input_df = pd.read_excel(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(uploaded_input_df)
        # Ensure the uploaded data matches the selected features
        if not all(feature in uploaded_input_df.columns for feature in selected_features):
            st.error("The uploaded file must contain all the selected features.")
            st.stop()
        uploaded_input_df = uploaded_input_df[selected_features]

# Decide which input_df to use
if 'uploaded_input_df' in locals():
    st.write("Using data from the uploaded Excel file for prediction.")
    input_df = uploaded_input_df
else:
    st.write("Using manually entered data for prediction.")
    input_df = manual_input_df


# Standardize the input data
if 'input_df' in locals():
    try:
        input_scaled = scaler.transform(input_df)
    except ValueError as e:
        st.error(f"Error: {e}")
        st.stop()
        

# Ensure input data has been processed and scaled
if 'input_scaled' in locals():
    if st.button("Predict"):
        predictions = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Display individual predictions
        for i, prediction in enumerate(predictions):
            if prediction == 1:
                st.success(f"Patient {i+1}: **NSCLC Patient**")
            else:
                st.success(f"Patient {i+1}: **Healthy Patient**")
            st.write(f"Prediction Confidence: {prediction_proba[i][prediction] * 100:.2f}%")

        # Show input data with predictions
        st.write("---")
        results = input_df.copy()
        results["Prediction"] = ["NSCLC" if p == 1 else "Healthy" for p in predictions]
        results["Confidence (%)"] = prediction_proba.max(axis=1) * 100
        st.write("Results with Predictions:")
        st.dataframe(results)

        # Option to download predictions
        csv = results.to_csv(index=False)
        st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")

        # Feature Importance Visualization
        st.write("---")
        st.write("Feature Importance (based on model coefficients):")
        feature_importances = model.coef_[0]  # Coefficients for the first (and only) class
        importance_df = pd.DataFrame({
            "Feature": selected_features,
            "Importance": np.abs(feature_importances)  # Use absolute values of coefficients
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature"))

        # SHAP Explanation
        st.write("---")
        st.write("SHAP Explanations for Feature Contributions:")

        explainer = shap.LinearExplainer(model, input_scaled)
        shap_values = explainer.shap_values(input_scaled)

        shap.initjs()
        for i in range(len(input_df)):
            st.write(f"Patient {i+1}:")
            shap_fig = shap.force_plot(
                explainer.expected_value,
                shap_values[i],
                input_df.iloc[i, :],
                matplotlib=True,
                show=False
            )
            st.pyplot(shap_fig, bbox_inches='tight')
else:
    st.warning("Please provide valid input data to generate predictions and SHAP explanations.")
