from src.prompts import SKILL_GAP_ANALYSIS_PROMPT
from src.data_prep.jd import parse_job_description
from src.data_prep.resume import parse_resume
from src.data_prep.utils import extract_text_from_pdf
from src.llm.langchain_genai import llm as llm
from src.cluster_skills import cluster_skills
from src.utils import extract_pydantic_elements
from src.md2table import extract_table_from_markdown


def penalize_passive_skills(df):
    # Define the penalization weights for each column category
    penalization_weights = {
        "Award": 0.9,
        "Publication": 0.8,
        "Project": 0.7,
        "Certification": 0.6,
    }

    # Apply penalization based on the column categories
    for col in df.columns[1:]:
        if "award" in col.lower():
            df[col] *= penalization_weights["Award"]
        elif "publication" in col.lower():
            df[col] *= penalization_weights["Publication"]
        elif "project" in col.lower():
            df[col] *= penalization_weights["Project"]
        elif "certification" in col.lower():
            df[col] *= penalization_weights["Certification"]
    return df


def extract_must_have_skills(jd_file):
    jd_text = extract_text_from_pdf(jd_file)
    parsed_jd = parse_job_description(jd_text)
    must_skill = [x.skill for x in parsed_jd.must_have_skills]
    return must_skill


def llm_gap_scores(
    job_title, must_skill, projects=None, certifications=None, publications=None, academic_awards=None
):
    prompt = SKILL_GAP_ANALYSIS_PROMPT.format(
        role=job_title,
        must_have_skills=must_skill,
        projects=extract_pydantic_elements(projects),
        certifications=extract_pydantic_elements(certifications),
        publications=extract_pydantic_elements(publications),
        awards=extract_pydantic_elements(academic_awards),
    )
    response = llm.invoke(input=prompt, generation_config=dict(temperature=0.3))
    df = extract_table_from_markdown(response.content)
    df = penalize_passive_skills(df)
    df_skills = df.set_index("Must-Have Skill").sum(1)
    df_count = (df.set_index("Must-Have Skill") > 0).sum(1)
    df_scores = (df_skills / (df_count + 1)).sort_values()
    return df_scores, df


def generate_gap_matrix(resume_file, jd_file, jd_text=None):
    if jd_text is not None:
        jd_text = extract_text_from_pdf(jd_file)
    parsed_jd = parse_job_description(jd_text)
    resume_text = extract_text_from_pdf(resume_file)
    parsed_resume = parse_resume(resume_text)
    parsed_jd.must_have_skills = [(x if isinstance(x, str) else x.skill) for x in parsed_jd.must_have_skills]
    prompt = SKILL_GAP_ANALYSIS_PROMPT.format(
        role=parsed_jd.job_title,
        must_have_skills=parsed_jd.must_have_skills,
        projects=extract_pydantic_elements(parsed_resume.projects),
        certifications=extract_pydantic_elements(parsed_resume.certifications),
        publications=extract_pydantic_elements(parsed_resume.publications),
        awards=extract_pydantic_elements(parsed_resume.academic_awards),
    )
    response = llm.invoke(input=prompt, generation_config=dict(temperature=0.3))
    df = extract_table_from_markdown(response.content)
    df = penalize_passive_skills(df)
    df_skills = df.set_index("Must-Have Skill").sum(1)
    df_count = (df.set_index("Must-Have Skill") > 0).sum(1)
    df_scores = (df_skills / (df_count + 1)).sort_values()
    # print()
    return df_scores, df, parsed_resume, parsed_jd


if __name__ == "__main__":
    generate_gap_matrix(
        resume_file="data/raw/resume/08082024cv (1).pdf",
        jd_file="data/raw/job_desc/AI-ML Engineer.pdf",
    )
