import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load datasets
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")
degree_df = pd.read_csv("degree_progress.csv")

# Merge degree progress
student_df = pd.merge(student_df, degree_df, on="student_id", how="left")

# Calculate degree progress
student_df["degree_progress_pct"] = (student_df["credits_completed"] / student_df["total_credits_required"]) * 100

# Flag behind schedule
def is_behind_schedule(row):
    expected_progress = (row["current_semester"] / 8) * 100  # assuming 8 semesters
    return "Yes" if row["degree_progress_pct"] < expected_progress - 10 else "No"

student_df["Behind Schedule"] = student_df.apply(is_behind_schedule, axis=1)

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

# Load LLM model from HuggingFace
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
llm = HuggingFacePipeline(pipeline=pipe)

# Conversation memory
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

# Custom logic for degree questions
def check_credit_query(user_input):
    user_input = user_input.lower()
    if "credits" in user_input and "graduate" in user_input and "student" in user_input:
        for sid in student_df["student_id"]:
            if sid.lower() in user_input:
                row = student_df[student_df["student_id"] == sid].iloc[0]
                needed = row["total_credits_required"] - row["credits_completed"]
                return f"Student {sid} needs {needed} more credits to graduate."
    return None

# Streamlit UI
st.set_page_config(page_title="Student Risk & Degree Progress Dashboard", layout="wide")
st.title("🎓 Student Risk & Degree Progress Dashboard")

role = st.selectbox("Select your role:", ["advisor", "chair"])
user_id = st.text_input(f"Enter your {role} ID:")

if user_id:
    if role == "advisor":
        allowed_students = mapping_df[mapping_df["advisor_id"] == user_id]["student_id"].tolist()
    else:
        allowed_students = mapping_df[mapping_df["program_chair_id"] == user_id]["student_id"].tolist()

    filtered_df = student_df[student_df["student_id"].isin(allowed_students)]

    if not filtered_df.empty:
        st.subheader("📊 Predicted Risk & Degree Progress")
        st.dataframe(filtered_df[["student_id", "Predicted Risk", "Risk Reason", "degree_progress_pct", "Behind Schedule", "Study Tips"]])

        st.markdown("### 📈 Risk Level Distribution")
        st.plotly_chart(px.pie(filtered_df, names="Predicted Risk", title="Risk Level Distribution"))

        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Student Summary", data=csv, file_name="student_summary.csv", mime='text/csv')

        st.markdown("### 🔍 Detailed Student View")
        for _, row in filtered_df.iterrows():
            with st.expander(f"Student ID: {row['student_id']}"):
                st.write({
                    "Risk": row["Predicted Risk"],
                    "Risk Reason": row["Risk Reason"],
                    "Credits Completed": row["credits_completed"],
                    "Total Required": row["total_credits_required"],
                    "Degree Progress": f"{row['degree_progress_pct']:.2f}%",
                    "Behind Schedule": row["Behind Schedule"],
                    "Study Tips": row["Study Tips"]
                })

        st.markdown("### 💬 Ask the Advisor Bot")
        user_input = st.text_input("Ask a question about a student or academic advising:")
        if user_input:
            credit_response = check_credit_query(user_input)
            if credit_response:
                st.success(credit_response)
            else:
                response = conversation.predict(input=user_input)
                st.success(response)
    else:
        st.warning("No students found for this user ID.")
