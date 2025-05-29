import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# For memory and escalation (LangChain + OpenAI)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import openai

# Load datasets
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")

# Features used
features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]

# KMeans clustering
X = student_df[features]
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
student_df["cluster"] = clusters

# Assign risk levels to clusters
cluster_centers = kmeans.cluster_centers_
risk_scores = [
    (1 - center[0]) + (1 - center[1]/4.0) + (1 - center[2]) + (1 - center[3])
    for center in cluster_centers
]
sorted_clusters = np.argsort(risk_scores)[::-1]
risk_labels = ["High", "Medium", "Low"]
cluster_to_risk = {cluster: risk_labels[rank] for rank, cluster in enumerate(sorted_clusters)}
student_df["Predicted Risk"] = student_df["cluster"].map(cluster_to_risk)

# Add Risk Reason and Study Tips
def generate_risk_reason_and_tips(row):
    reasons, tips = [], []
    if row["attendance_rate"] < 0.75:
        reasons.append(f"Low attendance ({row['attendance_rate']:.0%})")
        tips.append("Try setting alarms or calendar reminders to attend classes.")
    if row["gpa"] < 2.0:
        reasons.append(f"Low GPA ({row['gpa']:.2f})")
        tips.append("Seek tutoring or review foundational materials regularly.")
    if row["assignment_completion"] < 0.7:
        reasons.append(f"Incomplete assignments ({row['assignment_completion']:.0%})")
        tips.append("Create a to-do list and break assignments into smaller tasks.")
    if row["lms_activity"] < 0.5:
        reasons.append(f"Low LMS activity ({row['lms_activity']:.0%})")
        tips.append("Check LMS daily for updates and participate in forums.")
    return " & ".join(reasons) or "No major concerns", tips or ["Keep up the good work!"]

student_df[["Risk Reason", "Study Tips"]] = student_df.apply(
    lambda row: pd.Series(generate_risk_reason_and_tips(row)), axis=1
)

# Predictive Advising Tags
def predictive_flags(row):
    if row["attendance_rate"] < 0.5 or row["assignment_completion"] < 0.4:
        return "🚨 High Risk of Dropout"
    elif row["gpa"] < 1.5:
        return "⚠️ Needs Academic Counseling"
    return "✅ Stable"

student_df["Predictive Flag"] = student_df.apply(predictive_flags, axis=1)

# Streamlit App
st.set_page_config(page_title="Advisor Assistant with Memory", layout="wide")
st.title("🎓 Smart Advisor Assistant with Risk Prediction and Personalized Support")

# Role and Login
role = st.selectbox("Select your role:", ["advisor", "chair"])
user_id = st.text_input(f"Enter your {role} ID:")

# LangChain memory-enabled chat (simulated context-aware assistant)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
    st.session_state.chat_chain = ConversationChain(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        memory=st.session_state.memory
    )

st.subheader("💬 Ask Your Academic Assistant")
query = st.text_input("Type your question here:")
if query:
    response = st.session_state.chat_chain.run(query)
    st.markdown(f"**Assistant:** {response}")

    # Fallback to human advisor
    if "not sure" in response.lower() or "contact" in response.lower():
        st.info("🔁 Transferring to a human advisor for detailed support...")
        st.write(f"Context passed to human: `{query}`")

# Filter based on advisor or chair
if user_id:
    allowed_students = mapping_df[mapping_df[f"{role}_id"] == user_id]["student_id"].tolist()
    filtered_df = student_df[student_df["student_id"].isin(allowed_students)]

    if not filtered_df.empty:
        st.subheader("📊 Predicted Risk for Assigned Students")
        st.dataframe(filtered_df[["student_id"] + features + ["Predicted Risk", "Risk Reason", "Predictive Flag"]])

        # Pie Chart
        fig = px.pie(filtered_df, names="Predicted Risk", title="Risk Level Distribution")
        st.plotly_chart(fig)

        # Download
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", data=csv, file_name="student_risk_predictions.csv")

        # Detailed View
        st.markdown("### 🔍 Detailed View with Personalized Advice")
        for _, row in filtered_df.iterrows():
            with st.expander(f"Student ID: {row['student_id']}"):
                st.write({col: row[col] for col in features + ['Predicted Risk', 'Risk Reason', 'Predictive Flag']})
                st.markdown("**📝 Personalized Study Tips:**")
                for tip in row["Study Tips"]:
                    st.markdown(f"- {tip}")
    else:
        st.warning("No students found for your ID.")
