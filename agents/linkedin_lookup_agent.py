from langchain import PromptTemplate
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """given the full name {name} I want you to get  me a LinkedIn to their linkedin profile page.
     Your answer should contain only a URL"""
    tools_for_agent = [
        Tool(
            name="Crawl google 4 LinkedIn profile page",
            func=get_profile_url,
            description="useful for when you need to get the linkedin profile",
        )
    ]
    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    prompt_template = PromptTemplate(template=template, input_variables=["name"])
    linkedin_profile_url = agent.run(prompt_template.format_prompt(name=name))
    return linkedin_profile_url
