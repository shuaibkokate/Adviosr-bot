import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Load Datasets ---
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")

# --- Define feature columns ---
features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]

# --- Risk level generation logic ---
def generate_risk(row):
    score = (
        row["attendance_rate"] * 0.3 +
        row["gpa"] * 25 +
        row["assignment_completion"] * 0.2 +
        row["lms_activity"] * 0.3
    )
    if score >= 200:
        return "Low"
    elif score >= 140:
        return "Medium"
    else:
        return "High"

# Apply risk calculation
student_df["risk_level"] = student_df.apply(generate_risk, axis=1)

# --- Prepare data for training ---
label_map = {"Low": 0, "Medium": 1, "High": 2}
inverse_label_map = {v: k for k, v in label_map.items()}

X = student_df[features]
y = student_df["risk_level"].map(label_map)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy calculation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- Streamlit UI ---
st.set_page_config(page_title="Student Risk Predictor", layout="wide")
st.title("🎓 Student Risk Prediction Dashboard")

# ✅ Show model accuracy
st.markdown("### ✅ Model Accuracy (based on internal test data)")
st.metric(label="Accuracy", value=f"{accuracy * 100:.2f}%", delta=None)

# --- Role-Based Access Control ---
role = st.selectbox("Select your role:", ["advisor", "chair"])
user_id = st.text_input(f"Enter your {role} ID:")

if user_id:
    if role == "advisor":
        allowed_students = mapping_df[mapping_df["advisor_id"] == user_id]["student_id"].tolist()
    else:
        allowed_students = mapping_df[mapping_df["program_chair_id"] == user_id]["student_id"].tolist()

    filtered_df = student_df[student_df["student_id"].isin(allowed_students)]

    if not filtered_df.empty:
        X_filtered = filtered_df[features]
        predicted_risk = model.predict(X_filtered)
        filtered_df["Predicted Risk"] = [inverse_label_map[p] for p in predicted_risk]

        st.subheader("📊 Predicted Risk for Assigned Students")
        st.dataframe(filtered_df[["student_id"] + features + ["Predicted Risk"]])
    else:
        st.warning("No students found for this user ID.")
