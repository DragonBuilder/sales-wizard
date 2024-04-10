import os
import sys

from dotenv import load_dotenv
import requests

from bs4 import BeautifulSoup

if not load_dotenv():
    print("Failed to load .env file")
    sys.exit(1)

print(os.environ["SERPER_API_KEY"])

# os.environ["SERPER_API_KEY"] = "Your Key"  # serper.dev API key
# os.environ["OPENAI_API_KEY"] = "Your Key"

from crewai import Agent
from crewai_tools import SerperDevTool, tool



# search_tool = SerperDevTool()

@tool("Search Tool")
def search_tool(query: str):
    '''
    Search the web using SERP API
    '''
    url = "https://serpapi.com/search"
    parameters = {
        "api_key": os.getenv("SERPER_API_KEY"),
        "q": query,
        "start": 0,
        "num": 1
    }
    response = requests.get(url, params=parameters)
    return response.json()

@tool("Web Page Scraper")
def web_page_scrape_tool(url: str):
    '''
    Scrape text from a web page
    '''
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()


# Creating a senior researcher agent with memory and verbose mode
sales_agent = Agent(
  role='Expert Sales Agent Specialising in Educational Courses',
  goal="""
Providing valuable insights on how to sell a course on {topic}

Course Details
  
{course_details}
""",
  verbose=True,
  memory=True,
  backstory=(
    """
You have
    """
  ),
#   tools=[search_tool, web_page_scrape_tool],
  allow_delegation=True
)

# Creating a writer agent with custom tools and delegation capability
writer = Agent(
  role='Writer',
  goal='Write ',
  verbose=True,
  memory=True,
  backstory=(
    "With a flair for simplifying complex topics, you craft"
    "engaging sales pitches that highlight key opportunities for your audience."
  ),
#   tools=[search_tool, web_page_scrape_tool],
  allow_delegation=False
)

from crewai import Task

# Research task
sales_research_task = Task(
  description=(
"""
Identify key selling points for the course. How the course will help.
Also address where the opportunities are and potential risks or objections.

{topic}

Course Details
  
{course_details}

"""
  ),
  expected_output='A comprehensive sales report addressing oppurtunities and potential risks or objections',
#   tools=[search_tool, web_page_scrape_tool],
  agent=sales_agent,
)

# Writing task with language model configuration
write_task = Task(
  description=(
    """
Compose an engaging sales pitch on {topic}.

Course Details
  
{course_details}

Focus on why the course is valuable for students and how it will help them achieve their career goals. Highlight key lessons, what students will learn, and potential
Article should be comprehensive. Highlight key selling points of the course and address any potential conerns or objections.
"""
  ),
  expected_output='''
  A comprehensive sales pitch on how to sell the course on {topic}

  Course Details
  
  {course_details}
  ''',
#   tools=[search_tool, web_page_scrape_tool],
  agent=writer,
  async_execution=False,
  output_file='data/interior-design-sales-pitch.md'  # Example of output customization
)

from crewai import Crew, Process

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[sales_agent, writer],
  tasks=[sales_research_task, write_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)

course_details = ""

with open("data/course.md", 'r') as f:
    course_details = f.read()

result = crew.kickoff(inputs={'topic': 'Interior designing course', 'course_details': course_details})
print(result)