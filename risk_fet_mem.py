import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

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
cluster_centers = kmeans.cluster_centers_

# Create a score to rank clusters by risk (lower score => higher risk)
risk_scores = []
for center in cluster_centers:
    score = (1 - center[features.index("attendance_rate")]) + \
            (1 - center[features.index("gpa")] / 4.0) + \
            (1 - center[features.index("assignment_completion")]) + \
            (1 - center[features.index("lms_activity")])
    risk_scores.append(score)

# Map clusters to risk labels based on score ranking
sorted_clusters = np.argsort(risk_scores)[::-1]  # Descending order by risk score

risk_labels = ["High", "Medium", "Low"]
cluster_to_risk = {}
for rank, cluster_idx in enumerate(sorted_clusters):
    cluster_to_risk[cluster_idx] = risk_labels[rank]

# Map cluster assignments to risk labels
student_df["Predicted Risk"] = student_df["cluster"].map(cluster_to_risk)

# Function to generate reason for risk level per student
def generate_risk_reason(row):
    reasons = []
    if row["attendance_rate"] < 0.75:
        reasons.append(f"Low attendance ({row['attendance_rate']:.0%})")
    if row["gpa"] < 2.0:
        reasons.append(f"Low GPA ({row['gpa']:.2f})")
    if row["assignment_completion"] < 0.7:
        reasons.append(f"Low assignment completion ({row['assignment_completion']:.0%})")
    if row["lms_activity"] < 0.5:
        reasons.append(f"Low LMS activity ({row['lms_activity']:.0%})")
    return " & ".join(reasons) if reasons else "No major concerns detected"

student_df["Risk Reason"] = student_df.apply(generate_risk_reason, axis=1)

# Personalized Study Tips
def generate_study_tips(row):
    tips = []
    if row["gpa"] < 2.0:
        tips.append("Consider attending tutoring sessions and reviewing foundational materials.")
    if row["attendance_rate"] < 0.75:
        tips.append("Try to improve attendance and engage more in class activities.")
    if row["assignment_completion"] < 0.7:
        tips.append("Focus on completing and submitting assignments on time.")
    if row["lms_activity"] < 0.5:
        tips.append("Increase your participation in online course materials.")
    return " ".join(tips) if tips else "Keep up the good work!"

student_df["Study Tips"] = student_df.apply(generate_study_tips, axis=1)

# Load LLM model from HuggingFace (NO OpenAI key required)
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
llm = HuggingFacePipeline(pipeline=pipe)

# Conversation memory
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

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

        st.markdown("### 💬 Ask the Advisor Bot")
        user_input = st.text_input("Ask a question about a student or general advising help:")
        if user_input:
            response = conversation.predict(input=user_input)
            st.success(response)
    else:
        st.warning("No students found for this user ID.")
