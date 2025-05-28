import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")

# Prepare model and encoder
features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]

# Only use rows where risk_level is not null for training
train_df = student_df.dropna(subset=["risk_level"])
X = train_df[features]
y = train_df["risk_level"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("🎓 Student Risk Prediction Dashboard")
st.markdown(f"**🧮 Model Accuracy:** `{accuracy * 100:.2f}%`")

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
        filtered_df["Predicted Risk"] = le.inverse_transform(predicted_risk)

        st.subheader("📊 Predicted Risk for Assigned Students")
        st.dataframe(filtered_df[["student_id"] + features + ["Predicted Risk"]])
    else:
        st.warning("No students found for this user ID.")
