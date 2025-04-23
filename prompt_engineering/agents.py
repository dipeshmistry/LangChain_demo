import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_anthropic import ChatAnthropic
from agent_tools import tools


load_dotenv()

def get_claude_agent():
    llm = ChatAnthropic(
        model="claude-3-opus-20240229",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent
