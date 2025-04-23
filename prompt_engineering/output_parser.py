from langchain.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel, Field
from typing import Any,List,Dict,Optional

class SwabhavTech(BaseModel):
    technologies: List = Field(description="Technologies Swabhav Techlabs Works with",title="technologies")
    # output_parser = PydanticOutputParser(pydantic_object=SwabhavTech)
    def to_dict(self)->Dict[str,Any]:
        return {"technologies":self.technologies}
    
    @classmethod
    def model_json_schema(cls)->Dict[str,Any]:
        return cls.schema()
    
swabhav_technology_parser = PydanticOutputParser(pydantic_object=SwabhavTech)

class Experience(BaseModel):
    company: str = Field(..., description="Name of the company")
    role: str = Field(..., description="Job title or role")
    duration: Optional[str] = Field(None, description="Duration of employment")

    @classmethod
    def model_json_schema(cls)->Dict[str,Any]:
        return cls.schema()

class ResumeInfo(BaseModel):
    name: str = Field(..., description="Candidate's full name")
    email: str = Field(..., description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    skills: List[str] = Field(..., description="List of technical or soft skills")
    education: List[str] = Field(..., description="List of educational qualifications")
    experience: List[Experience] = Field(..., description="Work experience details")
    projects: Optional[List[str]] = Field(None, description="List of personal or professional projects")
    certifications: Optional[List[str]] = Field(None, description="Certifications acquired by the candidate")

    @classmethod
    def model_json_schema(cls)->Dict[str,Any]:  
        return cls.schema()

resume_parser = PydanticOutputParser(pydantic_object=ResumeInfo)