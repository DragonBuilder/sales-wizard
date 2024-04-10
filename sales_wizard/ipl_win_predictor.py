import os
import sys

from dotenv import load_dotenv
import requests

from bs4 import BeautifulSoup

from langchain.agents import Tool as LCTool 
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain.utilities.serpapi import SerpAPIWrapper
from langchain_openai import ChatOpenAI


from crewai import Agent, Task
from crewai_tools import tool as crew_tool


if not load_dotenv():
    print("Failed to load .env file")
    sys.exit(1)

search = SerpAPIWrapper()

# Create and assign the search tool to an agent
search_tool = LCTool(
  name="Web Search",
  func=search.run,
  description="Search the web using SERP API",
)

llm = ChatOpenAI(model="gpt-4-turbo-preview")

llm = ChatOpenAI()

@crew_tool("website Scraper")
def webscraper_tool(url):
    """
    Scrape a page.
    """
    response = requests.get(url)
    
#     llm_response = llm.invoke(f"""
# You are given the html contents website.
# Extract out the contents verbatum with legible formatting
                              
# {response.text}
# """)
    
    return response.text

@crew_tool("HTML text parser")
def html_parser_tool(html_content):
    """
    Extract Text contents from the html
    """
    soup = BeautifulSoup(html_content)
    soup.get_text()

@crew_tool("LLM")
def llm_tool(query: str) -> str:
    """
    Reason, analyse and predict on data using an LLM
    """
    return llm.invoke(query).content


senior_sports_analyst = Agent(
    role="You are a passionate cricket fan and a senior analyst at a company that predicts the win ratio for teams in a cricket match.",
    goal="To analyse and predict the win odds of each team in the upcoming IPL match between {home_team} and {away_team} to be played on {scheduled_date}",
    backstory='''
You use historical stats, minute by minute match histories and player's form to determine the winning odds of a team.
''',
    # tools = [],
    allow_delegation=True,
    verbose=True,
    memory=True,
)

sports_analyst = Agent(
    role="You are a passionate cricket fan and a professional cricket analyst.",
    goal="Discover insights on a IPL match, team or player.",
    backstory='''
You are a data and facts based analyst.
You base your judgments on data and reasoning.
You look at other analysis, news channels as well.
''',
    allow_delegation=False,
    tools = [search_tool, webscraper_tool, html_parser_tool],
    verbose=True,
    memory=True,
)

win_prediction_task = Task(
    description="""
Predict the win odds of each team in the upcoming IPL match between {home_team} and {away_team} to be played on {scheduled_date}.
""",
    agent=senior_sports_analyst,
    expected_output="The odds of winning for each of the team with reason.",
    output_file='data/RRvGT.md',
    # tools=[llm_tool]
)

from crewai import Crew, Process

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[senior_sports_analyst, sports_analyst],
  tasks=[
      win_prediction_task
  ],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)

result = crew.kickoff({"home_team": "RR", "away_team": "GT", "scheduled_date": "Apr 10, 2024"})