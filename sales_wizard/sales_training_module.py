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
sales_trainer_agent = Agent(
  role='Expert Sales Agent Specialising in Educational Courses',
  goal="""
Providing valuable insights on how to design a training program to sell a course on {topic}.

The course details as follows:

{course_details}

""",
  verbose=True,
  memory=True,
  backstory=(
    """
You have been working as a sales agent for 10 years, mostly selling educational courses. 
You have been a successful salesperson due to your ability to understand customer needs deeply and suggest why a course will help them achieve their goals. 
You also have a knack for addressing potential objections upfront in a non-confrontational manner.
More recently you have been selling a course on {topic}. The course details as follows: 

{course_details}
"""
  ),
#   tools=[search_tool, web_page_scrape_tool],
  allow_delegation=False
)

# Creating a writer agent with custom tools and delegation capability
writer = Agent(
  role='Writer',
  goal='''
Write engaging training sessions on how to effectively sell educational courses.
''',
  verbose=True,
  memory=True,
  backstory=(
"""
You are an expert writer.
You have also dabbled in teaching and training. 
While education is your passion, you recognize that selling courses also requires certain skills.
You have designed and published various online courses yourself, so you understand what qualities make a course successful and how it can truly benefit learners.
"""
  ),
#   tools=[search_tool, web_page_scrape_tool],
  allow_delegation=False
)

from crewai import Task

# Research task
sales_agent_skill_research_task = Task(
  description=(
"""
Identify key skills required to be an effective salesperson through research. 
Analyze recent trends in educational sales and common objections or concerns customers may have. 
Suggest strategies to overcome potential objections.
"""
  ),
  expected_output='A research paper on what makes an effective salesperson, recent trends in educational sales, and common objections or concerns customers may have along with strategies to address them',
#   tools=[search_tool, web_page_scrape_tool],
  agent=sales_trainer_agent,
)

course_selling_strategy_task = Task(
    description=(
"""
Identify sales strategies to sell the course. 
Consider different kinds of customers and come up with strategies to address their concerns. 
If certain customers are obviously don't persist them to buy.
The stragies should suggest workflows to identify if the customer is really a good fit or eagerly looking for a course in {topic}.
The strategy should then continue with ways to engage the customer and how to convert them. 
"""
  ),
  expected_output='''
A comprehensive list of sales strategies for the course given below:

{course_details}

''',
#   tools=[search_tool, web_page_scrape_tool],
  agent=sales_trainer_agent,
  # context=[sales_agent_skill_research_task]
)



# Writing task with language model configuration
training_programme_strategy_task = Task(
  description=(
    """
Design an effective training programme to provide sales training agents to sell a course on {topic}. The course details are given below:

{course_details}

The training will be given to sales agents with various backgrounds, such as

1. Recent graduates
2. People who have taught the course but lack sales experience.
3. People who have taught educational courses lack sales experience.
4. People with too casual mannerisms which could be considered rude by the client.
"""
  ),
  expected_output='''
A detailed training programme to that explains what makes an effective sales agent and provides strategies to do sales calls.
  ''',
#   tools=[search_tool, web_page_scrape_tool],
  agent=writer,
  async_execution=False,
  output_file='data/sales-agent-training-strategy.md',  # Example of output customization
  contexts=[sales_agent_skill_research_task, course_selling_strategy_task]
)



training_programme_module_creation_task = Task(
  description=(
    """
Design effective training programme modules to provide sales training agents on day to day workflow with activities. The course details are given below:

{course_details}

The training will be given to sales agents with various backgrounds, such as

1. Recent graduates
2. People who have taught the course but lack sales experience.
3. People who have taught educational courses lack sales experience.
4. People with too casual mannerisms which could be considered rude by the client.
"""
  ),
  expected_output='''
A detailed training programme to that explains what makes an effective sales agent and provides strategies to do sales calls.
  ''',
#   tools=[search_tool, web_page_scrape_tool],
  agent=writer,
  async_execution=False,
  output_file='data/sales-agent-training-module3.md',  # Example of output customization
  # contexts=[sales_agent_skill_research_task, course_selling_strategy_task]
)

from crewai import Crew, Process

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[sales_trainer_agent, writer],
  tasks=[
      sales_agent_skill_research_task, 
      course_selling_strategy_task, 
      training_programme_strategy_task, 
      training_programme_module_creation_task
  ],
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