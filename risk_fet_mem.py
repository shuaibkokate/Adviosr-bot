import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px
from transformers import pipeline
from langchain_community.chat_models import ChatHuggingFace
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load data
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")

# Define features for clustering
features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]
X = student_df[features]

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
student_df["cluster"] = clusters

# Determine cluster risk levels
cluster_centers = kmeans.cluster_centers_
risk_scores = []
for center in cluster_centers:
    score = (1 - center[features.index("attendance_rate")]) + \
            (1 - center[features.index("gpa")] / 4.0) + \
            (1 - center[features.index("assignment_completion")]) + \
            (1 - center[features.index("lms_activity")])
    risk_scores.append(score)

sorted_clusters = np.argsort(risk_scores)[::-1]
risk_labels = ["High", "Medium", "Low"]
cluster_to_risk = {cluster: risk_labels[i] for i, cluster in enumerate(sorted_clusters)}
student_df["Predicted Risk"] = student_df["cluster"].map(cluster_to_risk)

# Risk reasons and personalized tips
def generate_risk_reason(row):
    reasons = []
    tips = []
    if row["attendance_rate"] < 0.75:
        reasons.append(f"Low attendance ({row['attendance_rate']:.0%})")
        tips.append("Attend all lectures and monitor your attendance weekly.")
    if row["gpa"] < 2.0:
        reasons.append(f"Low GPA ({row['gpa']:.2f})")
        tips.append("Seek academic advising and join peer study groups.")
    if row["assignment_completion"] < 0.7:
        reasons.append(f"Low assignment completion ({row['assignment_completion']:.0%})")
        tips.append("Use a planner to meet deadlines and ask for help when stuck.")
    if row["lms_activity"] < 0.5:
        reasons.append(f"Low LMS activity ({row['lms_activity']:.0%})")
        tips.append("Log into LMS daily to stay updated on coursework.")
    return (" & ".join(reasons), "\n".join(tips))

student_df[["Risk Reason", "Study Tips"]] = student_df.apply(lambda row: pd.Series(generate_risk_reason(row)), axis=1)

# Initialize Hugging Face model for conversational memory and prediction
txt_gen_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")
llm = ChatHuggingFace(pipeline=txt_gen_pipeline)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Streamlit UI
st.set_page_config(page_title="Student Risk Predictor (Clustering + Memory)", layout="wide")
st.title("🎓 Student Risk Prediction & Advising Dashboard")

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
        st.dataframe(filtered_df[["student_id"] + features + ["Predicted Risk", "Risk Reason", "Study Tips"]])

        st.markdown("### 📈 Risk Level Distribution")
        fig = px.pie(filtered_df, names="Predicted Risk", title="Risk Level Distribution")
        st.plotly_chart(fig)

        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", data=csv, file_name="student_risk_predictions.csv", mime='text/csv')

        st.markdown("### 🔍 Detailed Student View")
        for _, row in filtered_df.iterrows():
            with st.expander(f"Student ID: {row['student_id']}"):
                st.write({col: row[col] for col in features + ['Predicted Risk', 'Risk Reason', 'Study Tips']})

        st.markdown("### 🤖 Advisor Chatbot (with memory & prediction)")
        user_input = st.text_input("Ask a question or get advice:")
        if user_input:
            if "human" in user_input.lower():
                st.warning("Escalated to human advisor. Please check your email or schedule a meeting.")
            elif "predict" in user_input.lower():
                st.info("Predictive advising triggered. Checking for students at risk of missing deadlines or needing counseling...")
                high_risk = filtered_df[filtered_df["Predicted Risk"] == "High"]
                st.write("⚠️ At-risk students:", high_risk["student_id"].tolist())
            else:
                response = conversation.run(user_input)
                st.success(response)
    else:
        st.warning("No students found for this user ID.")
