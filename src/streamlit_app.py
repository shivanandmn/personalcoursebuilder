import streamlit as st
import pandas as pd
import os
os.system("pip install -e .")
from src.skill2resume_matrix import (
    generate_gap_matrix,
    llm_gap_scores,
)
from src.recommender import CourseRecommenderPipeline
from src.db.db_transact import DBTransaction
from src.db.storage import GCPStorage


# Streamlit UI
st.title("Job Recommendation System for Freshers")

st.sidebar.header("Upload Files")
job_desc_file = st.sidebar.file_uploader("Upload Job Description (PDF)", type=["pdf"])
resume_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])

db_transaction = DBTransaction(db_name="course_builder")
gcp_storage = GCPStorage(bucket_name="course_builder_dataset")

if st.sidebar.button("Submit"):
    if job_desc_file and resume_file:

        # Generate Gap Matrix
        gap_scores, df_gap, parsed_resume, parsed_jd = generate_gap_matrix(
            resume_file=resume_file, jd_file=job_desc_file
        )

        # Initialize Course Recommender Pipeline
        course_recommender = CourseRecommenderPipeline(
            "./data/raw/course/courses_103190.csv"
        )
        user_preferences = {
            "min_duration": 1,
            "max_duration": 40,
            "min_rating": 4.0,
            "language": "English",
            "top_n": 20,
        }

        # Get Course Recommendations
        rec_results = course_recommender.recommend_courses(
            parsed_jd.must_have_skills, user_preferences
        )
        df_sorted = rec_results["all_recommendations"]
        df_deduplicated = rec_results["dedup_recommendations"]

        certifications = df_deduplicated[["name", "description", "level"]].to_dict(
            "records"
        )
        improved_scores, df_improved = llm_gap_scores(
            job_title=parsed_jd.job_title,
            must_skill=parsed_jd.must_have_skills,
            certifications=certifications,
        )

        cleaned_courses = {
            k: k.replace("(Certificate)", "").strip()
            for k in df_improved.columns
            if (not k.startswith("Project Name") or not "Project" in k)
        }
        df_improved = df_improved.rename(columns=cleaned_courses)
        df_improved = df_improved[cleaned_courses.values()]
        courses_filtered = (
            df_improved.sum(0).reset_index().iloc[1:].sort_values(by=0, ascending=False)
        )

        # Update Recommendations with Additional Information
        rec_results.update(
            {
                "courses_filtered_after_dedup": courses_filtered,
                "initial_knowledge": gap_scores,
                "score_can_be_improved": improved_scores,
                "df_gap": df_gap,
                "parsed_jd": parsed_jd.model_dump_json(),
                "parsed_resume": parsed_resume.model_dump_json(),
            }
        )

        # Display Results in Card-Like UI
        st.header("Course Recommendations")
        st.subheader("Based on Most Knowledge Delta")
        for index, row in df_deduplicated.iterrows():
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #e0e0e0;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 15px;
                        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                    ">
                        <h3 style="color: #0073e6;">{row['name']}</h3>
                        <p><strong>Provider:</strong> {row['provider']}</p>
                        <p><strong>Level:</strong> {row['level'] if row['level'] != 'nan' else 'N/A'}</p>
                        <p><strong>Duration:</strong> {row['duration']}</p>
                        <p><strong>Description:</strong> {row['description']}</p>
                        <p><strong>Language:</strong> {row['language']}</p>
                        <p><strong>Rating:</strong> {row['avg_rating']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Display Metadata
        st.header("Meta Data")
        with st.expander("Initial Knowledge"):
            st.dataframe(gap_scores)

        with st.expander("Scores After Certification"):
            st.dataframe(improved_scores)

        with st.expander("Gap Dataframe"):
            st.dataframe(df_gap)

        st.header("Parsed Data")
        with st.expander("Parsed Job Description Data"):
            st.json(parsed_jd.model_dump_json())
        with st.expander("Parsed Resume Data"):
            st.json(parsed_resume.model_dump_json())

        jd_url = gcp_storage.upload_file(
            job_desc_file.name,
            destination_blob_name=job_desc_file.name,
            folder="job_description",
        )
        resume_url = gcp_storage.upload_file(
            resume_file.name,
            destination_blob_name=resume_file.name,
            folder_path="resume_dataset",
        )
        save_data = {
            k: (v.to_dict() if isinstance(v, pd.DataFrame) else v)
            for k, v in rec_results.items()
        }
        save_data["jd_url"] = jd_url
        save_data["resume_url"] = resume_url
        inserted_id = db_transaction.insert_one(
            save_data,
            collection="interactions_data",
        )
    else:
        st.warning("Please upload both Job Description and Resume files.")
