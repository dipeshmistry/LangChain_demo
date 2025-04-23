from langchain_openai import ChatOpenAI
from output_parser import swabhav_technology_parser, SwabhavTech
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import os
load_dotenv()
# print(os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(temperature=0.7)

template = """
Context : swabhav techlabs works with technologies like dotnet and golang for software development.

Task : {query}

instructions :
1. if you dont get the question from the context then say "I don't know"
\n {format_instructions}
"""
print(swabhav_technology_parser.get_format_instructions())
prompt = PromptTemplate(template=template, input_variables=["task"],partial_variables={"format_instructions":swabhav_technology_parser.get_format_instructions()})

chain = prompt | llm

query = "what technologies does swabhav work with?"
response = chain.invoke({"query": query})
print(response)
