from typing import List, Optional
from pydantic import BaseModel, Field, conint, constr
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate
from src.data_prep.utils import extract_text_from_pdf
from src.llm.langchain_genai import llm as llm
from langchain_core.output_parsers import PydanticOutputParser

# Pydantic models


# Enum to restrict project types
class ProjectType(str, Enum):
    PERSONAL = "personal"
    INTERNSHIP = "internship"
    COMPANY = "company"
    NOMENTION = "no-mention"
    OPENSOURCE = "open-source"


# Updated Pydantic models
class Project(BaseModel):
    """
    Represents a project that an individual has worked on.
    Fields:
        - title: The name of the project (Required).
        - description: A brief description of the project (Optional).
        - type: The classification of the project (e.g., personal, internship, company, no-mention) (Required).
    """

    title: str = Field(..., description="The title of the project.")
    description: Optional[str] = Field(
        description="Description of the project."
    )
    type: ProjectType = Field(
        "no-mention",
        description="Classify the project as 'personal'(project is done without any external help), 'internship' (project is done in the internship), 'company'(project is done in a company), 'no-mention'(project is not mentioned), 'open-source'(project is open-source contribution)",
    )


class Certification(BaseModel):
    """
    Represents a professional certification obtained by the individual.
    Fields:
        - name: The name of the certification (Required).
        - organization: The organization that issued the certification (Optional).
    """

    name: str = Field(..., description="The name of the certification.")
    organization: Optional[str] = Field(
        description="The organization that issued the certification."
    )


class Publication(BaseModel):
    """
    Represents a research publication authored by the individual.
    Fields:
        - title: The title of the research publication (Required).
        - journal: The journal where the publication was published (Optional).
        - year: The year the publication was released, limited between 1900 and 2100 (Optional).
    """

    title: str = Field(..., description="The title of the research publication.")
    journal: Optional[str] = Field(
        description="The journal where the publication was published."
    )
    year: Optional[conint(ge=1900, le=2100)] = Field(  # type: ignore
        description="The year the publication was released."
    )


class AcademicAward(BaseModel):
    """
    Represents an academic award received by the individual.
    Fields:
        - title: The title of the academic award (Required).
        - related_skills: A list of skills/tools/framework related to the award (Optional).
    """

    title: str = Field(..., description="The title of the academic award received.")
    related_skills: Optional[List[str]] = Field(
        description="A list of skills/tools/framework related to the award."
    )


class ResumeData(BaseModel):
    """
    Represents the complete parsed resume data of an individual.
    Fields:
        - skills: A list of technical skills/tools/framework mentioned in the resume (Defaults to an empty list).
        - certifications: A list of certifications earned by the individual (Optional, Defaults to an empty list).
        - projects: A list of projects the individual has worked on (Optional, Defaults to an empty list).
        - publications: A list of research publications by the individual (Optional, Defaults to an empty list).
        - academic_awards: A list of academic awards received by the individual (Optional, Defaults to an empty list).
    """

    skills: List[str] = Field(
        default_factory=list,
        description="A list of technical skills/tools/framework mentioned in the resume.",
    )
    certifications: Optional[List[Certification]] = Field(
        description="A list of certifications earned by the individual."
    )
    projects: Optional[List[Project]] = Field(
        description="A list of projects the individual has worked on."
    )
    publications: Optional[List[Publication]] = Field(
        description="A list of research publications by the individual."
    )
    academic_awards: Optional[List[AcademicAward]] = Field(
        description="A list of academic/technical awards received by the individual, don't consider co-curriculum or soft skill awards."
    )


# Set up a parser
parser = PydanticOutputParser(pydantic_object=ResumeData)

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


# Resume Parsing Pipeline
def parse_resume(resume_text: str) -> ResumeData:
    response = chain.invoke({"query": resume_text})
    return response


# Example usage
if __name__ == "__main__":
    sample_resume_text = extract_text_from_pdf(
        "data/raw/resume/JOEL PHILIP RESUME .pdf"
    )

    parsed_resume = parse_resume(sample_resume_text)
    print(parsed_resume.model_dump_json(indent=4))
    print()
