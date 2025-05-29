import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px

# Load datasets
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")

# Define features for clustering
features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]

X = student_df[features]

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster assignments to dataframe
student_df["cluster"] = clusters

# Determine cluster risk level by comparing cluster centers
# Assuming lower GPA & attendance = higher risk
cluster_centers = kmeans.cluster_centers_

# Create a score to rank clusters by risk (lower score => higher risk)
# For example, sum of inverse GPA, inverse attendance, etc.
risk_scores = []
for center in cluster_centers:
    # We invert GPA and attendance for risk scoring:
    score = (1 - center[features.index("attendance_rate")]) + \
            (1 - center[features.index("gpa")] / 4.0) + \
            (1 - center[features.index("assignment_completion")]) + \
            (1 - center[features.index("lms_activity")])
    risk_scores.append(score)

# Map clusters to risk labels based on score ranking
# Higher score = higher risk, so we sort descending
sorted_clusters = np.argsort(risk_scores)[::-1]

risk_labels = ["High", "Medium", "Low"]
cluster_to_risk = {}
for rank, cluster_idx in enumerate(sorted_clusters):
    cluster_to_risk[cluster_idx] = risk_labels[rank]

# Map cluster assignments to risk labels
student_df["Predicted Risk"] = student_df["cluster"].map(cluster_to_risk)

# Streamlit UI
st.set_page_config(page_title="Student Risk Predictor (Clustering)", layout="wide")
st.title("🎓 Student Risk Prediction Dashboard (Unsupervised)")

role = st.selectbox("Select your role:", ["advisor", "chair"])
user_id = st.text_input(f"Enter your {role} ID:")

if user_id:
    if role == "advisor":
        allowed_students = mapping_df[mapping_df["advisor_id"] == user_id]["student_id"].tolist()
    else:
        allowed_students = mapping_df[mapping_df["program_chair_id"] == user_id]["student_id"].tolist()

    filtered_df = student_df[student_df["student_id"].isin(allowed_students)]

    if not filtered_df.empty:
        st.subheader("📊 Predicted Risk for Assigned Students")
        st.dataframe(filtered_df[["student_id"] + features + ["Predicted Risk"]])

        # Pie chart
        st.markdown("### 📈 Risk Level Distribution")
        fig = px.pie(filtered_df, names="Predicted Risk", title="Risk Level Distribution")
        st.plotly_chart(fig)

        # Export CSV
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", data=csv, file_name="student_risk_predictions.csv", mime='text/csv')

        # Detailed View
        st.markdown("### 🔍 Detailed Student View")
        for _, row in filtered_df.iterrows():
            with st.expander(f"Student ID: {row['student_id']}"):
                st.write({col: row[col] for col in features + ['Predicted Risk']})
    else:
        st.warning("No students found for this user ID.")
