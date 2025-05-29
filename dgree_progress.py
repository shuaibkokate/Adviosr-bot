# Load additional dataset for degree progress
degree_df = pd.read_csv("degree_progress.csv")

# Merge with main student data
student_df = pd.merge(student_df, degree_df, on="student_id", how="left")

# Calculate progress percentage
student_df["degree_progress_percent"] = (student_df["credits_completed"] / student_df["total_credits_required"]) * 100
student_df["degree_progress_percent"] = student_df["degree_progress_percent"].round(2)

# Update UI
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
        st.subheader("📊 Predicted Risk & Degree Progress")
        st.dataframe(filtered_df[["student_id"] + features + ["Predicted Risk", "Risk Reason", "Study Tips", "degree_progress_percent"]])

        st.markdown("### 📈 Risk Level Distribution")
        fig = px.pie(filtered_df, names="Predicted Risk", title="Risk Level Distribution")
        st.plotly_chart(fig)

        st.markdown("### 🎯 Degree Progress Overview")
        for _, row in filtered_df.iterrows():
            with st.expander(f"Student ID: {row['student_id']}"):
                st.write({
                    "Attendance Rate": f"{row['attendance_rate']:.0%}",
                    "GPA": row["gpa"],
                    "Assignment Completion": f"{row['assignment_completion']:.0%}",
                    "LMS Activity": f"{row['lms_activity']:.0%}",
                    "Risk": row["Predicted Risk"],
                    "Risk Reason": row["Risk Reason"],
                    "Study Tips": row["Study Tips"],
                    "Credits Completed": row["credits_completed"],
                    "Total Credits Required": row["total_credits_required"],
                    "Degree Progress": f"{row['degree_progress_percent']}%",
                    "Core Courses Completed": row["core_courses_completed"],
                    "Electives Completed": row["electives_completed"]
                })
                st.progress(min(row['degree_progress_percent'] / 100, 1.0))

        st.markdown("### 💬 Ask the Advisor Bot")
        user_input = st.text_input("Ask a question about a student or general advising help:")
        if user_input:
            response = conversation.predict(input=user_input)
            st.success(response)

        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", data=csv, file_name="student_risk_predictions.csv", mime='text/csv')

    else:
        st.warning("No students found for this user ID.")
