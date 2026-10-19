from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from tools import scrape_url, web_search
import os

class WriterPromptOutput(BaseModel):
    introduction: str = Field(description="Introduction of the research topic")
    key_findings: List[str] = Field(description="key findings")
    conclusion: str = Field(description="conclusion of the research topic")
    sources: List[str] = Field(description="list of all urls from where you collect the information")

class CriticPromptOutput(BaseModel):
    score: int = Field(ge=0, le=10, description="score of the report")
    strength: List[str]
    areas_to_improve: List[str]
    one_line_verdict: str = Field(description="one liner verdict for the report")



model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"))
str_parser = StrOutputParser()
writer_pydantic_parser = PydanticOutputParser(pydantic_object=WriterPromptOutput)
critic_pydantic_parser = PydanticOutputParser(pydantic_object=CriticPromptOutput)

def build_search_agent():
    return create_agent(
        model=model, 
        tools=[web_search]
    )

def build_reader_agent():
    return create_agent(
        model=model,
        tools=[scrape_url]
    )

writer_prompt = PromptTemplate(
    template="""You are an expert research writer, write clear, structured and insightful reports"
    "Write a detailed research report on the topic below."
        Topic : {topic}
        Research : {research}

        Struture the report as: {structure}
        Be detailed, factual and professional""",
    input_variables=['topic', 'research'],
    partial_variables={"structure": writer_pydantic_parser.get_format_instructions()}
)

writer_chain = writer_prompt | model | str_parser

critic_prompt = PromptTemplate(
    template="""
        You are a sharp and constructive research critic, Be honest and specific.
        Review the research paper below and evaluate it strictly.
        Report: {report}
        Respond in the exact format: {structure}
    """,
    input_variables=['report'],
    partial_variables={"structure": critic_pydantic_parser.get_format_instructions()}
)

critic_chain = critic_prompt | model | str_parser