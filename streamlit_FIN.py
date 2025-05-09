import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load model and helpers ---
model = joblib.load("dropout_predictor_model.pkl")
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans.pkl")
with open("top_features.csv") as f:
    top_features = [line.strip() for line in f.readlines()]

categorical_cols = [
    "sex", "school", "Mjob", "Fjob", "reason", "guardian", "address", "famsize",
    "Pstatus", "schoolsup_mat", "famsup_mat", "paid_mat", "activities", "nursery",
    "higher", "internet", "romantic", "schoolsup_por", "famsup_por", "paid_por"
]
encoders = {col: joblib.load(f"encoders/{col}_encoder.pkl") for col in categorical_cols}

X_train_structure = pd.read_csv("X_train_structure.csv")
expected_columns = X_train_structure.columns.tolist()
expected_dtypes = X_train_structure.dtypes.to_dict()

# --- Helper to align input ---
def align_columns(input_df):
    input_df = input_df.copy()
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0 if np.issubdtype(expected_dtypes[col], np.number) else "unknown"
    input_df = input_df[expected_columns]
    for col in expected_columns:
        try:
            input_df[col] = input_df[col].astype(expected_dtypes[col])
        except Exception as e:
            st.warning(f"⚠️ Could not cast column '{col}': {e}")
    return input_df

# --- Cluster-based interventions ---
cluster_interventions = {
    1: "Focus on improving grades, increasing study time, and reducing absences.",
    2: "Reduce absences, eliminate course failures, and increase study time.",
    3: "Maintain high grades, sustain good study habits, and monitor health.",
}

# --- Streamlit UI ---
st.title("📊 InsightEd: School Dropout Prediction and Prevention")
st.write("Upload a CSV or Excel file containing student data to predict dropout risk in bulk and get intervention suggestions for high-risk students.")
st.markdown('Find the tutorial to fill the template here: [tinyurl.com/InsightEd](https://tinyurl.com/InsightEd)')

# --- Downloadable template for users ---
template_columns = expected_columns.copy()
if "student_id" not in template_columns:
    template_columns.insert(0, "student_id")
template_df = pd.DataFrame(columns=template_columns)
template_csv = template_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📄 Download Student Data Template",
    template_csv,
    "student_data_template.csv",
    "text/csv"
)

uploaded_file = st.file_uploader("Upload your student data file (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Try reading CSV with auto delimiter detection
        if uploaded_file.name.endswith(".csv"):
            content = uploaded_file.read().decode("utf-8")
            delimiter = ";" if ";" in content.splitlines()[0] else ","
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, delimiter=delimiter)
        else:
            df = pd.read_excel(uploaded_file)

        if "student_id" not in df.columns:
            st.error("❌ The file must include a 'student_id' column.")
        else:
            # Encode categorical columns
            for col in encoders:
                if col in df.columns:
                    try:
                        df[col] = encoders[col].transform(df[col])
                    except Exception as e:
                        st.warning(f"⚠️ Encoding failed for '{col}': {e}")

            # Align structure
            aligned_df = align_columns(df)

            # Predict
            predictions = model.predict(aligned_df)
            df["dropout_predicted"] = predictions

            num_dropouts = df["dropout_predicted"].sum()
            st.success(f"✅ {num_dropouts} students predicted to drop out.")

            # --- Cluster-based intervention suggestions for high-risk students ---
            high_risk_df = df[df["dropout_predicted"] == 1].copy()
            if not high_risk_df.empty:
                # Assign clusters using the top features, and shift to 1,2,3
                X_cluster = high_risk_df[top_features]
                X_scaled = scaler.transform(X_cluster)
                high_risk_df["cluster"] = kmeans.predict(X_scaled) + 1
                high_risk_df["intervention_suggestions"] = high_risk_df["cluster"].map(cluster_interventions)

                # Download only high-risk students
                csv = high_risk_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download High-Risk Students with Interventions",
                    csv,
                    "high_risk_with_interventions.csv",
                    "text/csv"
                )
            else:
                st.info("No high-risk students found in the uploaded data.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")