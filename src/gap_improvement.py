from src.skill2resume_matrix import (
    generate_gap_matrix,
    extract_must_have_skills,
    llm_gap_scores,
)
from src.recommender import CourseRecommenderPipeline

job_desc_file = "data/raw/job_desc/AI-ML Engineer.pdf"
resume_file = "data/raw/resume/08082024cv (1).pdf"

# Generate Gap Matrix
gap_scores, df_gap, parsed_resume, parsed_jd = generate_gap_matrix(
    resume_file=resume_file, jd_file=job_desc_file
)

# Initialize Course Recommender Pipeline
course_recommender = CourseRecommenderPipeline("./data/raw/course/courses_103190.csv")
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

certifications = df_deduplicated[["name", "description", "level"]].to_dict("records")
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
print(rec_results)
