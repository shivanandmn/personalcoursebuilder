from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate
from src.data_prep.utils import extract_text_from_pdf
from src.llm.langchain_genai import llm as llm
from langchain_core.output_parsers import PydanticOutputParser

# Pydantic models


# Enum to restrict employment types
class EmploymentType(str, Enum):
    FULLTIME = "full-time"
    PARTTIME = "part-time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"


class KeyResponsibility(BaseModel):
    """
    Represents a key responsibility for a given job.
    Fields:
        - description: A brief description of the responsibility (Required).
    """

    description: str = Field(
        ..., description="A brief description of the key responsibility."
    )


class JobQualification(BaseModel):
    """
    Represents a qualification required for the job.
    Fields:
        - qualification: A description of the qualification required, other than education qualification (Required).
    """

    qualification: str = Field(
        ..., description="A description of the job qualification."
    )


class SkillSet(BaseModel):
    """
    Represents a set of skills for the job.
    Fields:
        - skill: The name of the required skill (Required).
    """

    skill: str = Field(..., description="The name of the required skill.")


class EducationQualification(BaseModel):
    """
    Represents an education qualification required for the job.
    Fields:
        - degree: The name of the degree or education level required (Required).
        - field_of_study: The field of study related to the degree (Optional).
    """

    degree: str = Field(
        None, description="The name of the degree or education level required."
    )
    field_of_study: Optional[str] = Field(
        description="The field of study related to the degree."
    )


class JobDescriptionData(BaseModel):
    """
    Represents the complete parsed job description data.
    Fields:
        - job_title: The title of the job position (Optional).
        - employment_type: The employment type for the job (Optional).
        - key_responsibilities: A list of key responsibilities (Optional, Defaults to an empty list).
        - qualifications: A list of job qualifications (Optional, Defaults to an empty list).
        - education_requirements: A list of education qualifications required for the job (Optional, Defaults to an empty list).
        - must_have_skills: A list of required technical skills (Optional, Defaults to an empty list).
        - bonus_skills: A list of bonus technical skills (Optional, Defaults to an empty list).
    """

    job_title: Optional[str] = Field(description="The title of the job position.")
    employment_type: Optional[EmploymentType] = Field(
        description="The type of employment, such as 'full-time', 'part-time', 'contract', etc."
    )
    key_responsibilities: Optional[List[KeyResponsibility]] = Field(
        default_factory=list, description="A list of key responsibilities for the job."
    )
    qualifications: Optional[List[JobQualification]] = Field(
        default_factory=list,
        description="A list of qualifications required for the job.",
    )
    education_requirements: Optional[List[EducationQualification]] = Field(
        default_factory=list,
        description="A list of education qualifications required for the job.",
    )
    must_have_skills: Optional[List[SkillSet]| List[str]] = Field(
        default_factory=list,
        description="A list of required/must/should technical skills, don't mention soft skills like teamwork, communication",
    )
    bonus_skills: Optional[List[SkillSet] | List[str]] = Field(
        default_factory=list,
        description="A list of bonus(good to have) technical skills.",
    )


# Set up a parser
parser = PydanticOutputParser(pydantic_object=JobDescriptionData)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",
        ),
        ("human", "{query}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Create the chain
chain = prompt | llm | parser


# Job Description Parsing Pipeline
def parse_job_description(job_description_text: str) -> JobDescriptionData:
    response = chain.invoke({"query": job_description_text})
    return response


# Example usage
if __name__ == "__main__":
    sample_job_description_text = extract_text_from_pdf(
        "data/raw/job_desc/AI-ML Engineer.pdf"
    )

    parsed_job_description = parse_job_description(sample_job_description_text)
    print(parsed_job_description.model_dump_json(indent=4))
    print()
