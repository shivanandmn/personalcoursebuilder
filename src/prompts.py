SKILL_GAP_ANALYSIS_PROMPT = """\
Your are expert {role}, You are tasked with performing a detailed Skill Gap Analysis between a job description's required skills and a candidate's resume. Your job is to map the must-have skills from the job description to relevant elements in the resume (projects, certifications, publications, awards) and generate a matrix showing the mapping.

### Step 1: Understand the Must-Have Skills
Here is a list of must-have skills extracted from the job description: {must_have_skills}

### Step 2: Parse the Candidate's Profile Elements
Here are the key elements from the candidate's resume that need to be mapped to these skills, if they are present otherwise don't mention it.
- **Projects**: \n{projects}
- **Certifications**: \n{certifications}
- **Publications**: \n{publications}
- **Awards**: \n{awards}

### Step 3: Skill Mapping and Scoring
For each skill, map it to the relevant profile elements based on the text provided. Use the following guidelines:
- Consider **direct matches** (e.g., if the resume explicitly mentions the skill).
- Consider **indirect matches** where the skill is demonstrated through context (e.g., a project that used techniques related to the skill, even if the skill is not mentioned by name).
- If the skill is only **partially covered**, assign a partial score (e.g., 0.5 for moderate coverage).
- If the skill is **fully covered**, assign a full score of 1.
- If the skill is **not covered at all**, assign a score of 0.
- Always keep the default score of 0 if profile elements not present.

### Step 4: Output Structure
Output a matrix with the following structure:

| Must-Have Skill   | Award Name X (Award) | Publication Name X (Publication) | Project Name X (Project) | Project Name Y (Project) | Certificate Name X (Certificate) |
|-------------------|----------------------|----------------------------------|--------------------------|--------------------------|----------------------------------|
| Python            | 0                    | 0                                | 1                        | 0                        | 0                                |
| Machine Learning  | 0                    | 0                                | 0.2                      | 0                        | 0.5                              |


Provide the reasoning behind each score. For example:
- "Python" in Project-1 gets a score of 1 because the project explicitly mentions using Python for development.
- "Machine Learning" in Certification-1 gets a score of 0.5 because the certification includes some ML-related topics, but not in depth.
- If profile element is not present then default value will be 0.
- If Projects or certificate or any other things are not present then don't mention it at all, don't mention it as Project Name X or any other way.

### Step 5: Additional Considerations
- Use the context of the text to infer whether a skill is demonstrated, even if it's not explicitly mentioned.
- Make sure to handle skills that are partially demonstrated or related indirectly.
- Output the matrix with scores in a well-organized table format.
- Summarize the strengths and skill gaps after creating the matrix.

"""
