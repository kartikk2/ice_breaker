from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from agents.linkedin_lookup_agent import lookup
from third_parties.linkedin import scrape_linkedin_profile

information = """
    Mark Elliot Zuckerberg (/ˈzʌkərbɜːrɡ/; born May 14, 1984), also known colloquially as Zuck, is an American business magnate, computer programmer, internet entrepreneur and philanthropist. He is known for co-founding the social media website Facebook and its parent company Meta Platforms (formerly Facebook, Inc.), of which he is the executive chairman, chief executive officer, and controlling shareholder.[1][2]
    Zuckerberg attended Harvard University, where he launched Facebook in February 2004 with his roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz, and Chris Hughes. Originally launched to select college campuses, the site expanded rapidly and eventually beyond colleges, reaching one billion users in 2012. Zuckerberg took the company public in May 2012 with majority shares. In 2007, at age 23, he became the world's youngest self-made billionaire. He has used his funds to organize multiple philanthropic endeavors, including the establishment of the Chan Zuckerberg Initiative.
    Zuckerberg has been listed as one of the most influential people in the world on four occasions in 2008, 2011, 2016 and 2019 respectively and nominated as a finalist in 2009, 2012, 2014, 2015, 2017, 2018, 2020, 2021 and 2022. He was named the Person of the Year by Time magazine in 2010, the same year when Facebook eclipsed more than half a billion users.[3][4][5] In December 2016, Zuckerberg was ranked tenth on Forbes list of The World's Most Powerful People.[6] In the Forbes 400 list of wealthiest Americans in 2022, he was ranked 11th with a wealth of $57.7 billion, down from his status as the third-richest American in 2021 with a net worth of $134.5 billion. As of May 2023, Zuckerberg's net worth was estimated at $85.0 billion according to the Forbes Real Time Billionaires, making him the 14th richest person in the world.[7] A film depicting Zuckerberg's early career, legal troubles and initial success with the site, The Social Network, was released in 2010 and won multiple Academy Awards.
"""


if __name__ == "__main__":
    print("Hello langhchain")

    linkedin_profile_url = lookup(name="Eden Marco Udemy")

    summaryTemplate = """
        given the Linkedin information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summaryPromptTemplate = PromptTemplate(
        input_variables=["information"], template=summaryTemplate
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summaryPromptTemplate)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    # print(linkedin_data)

    print(chain.run(information=linkedin_data))
