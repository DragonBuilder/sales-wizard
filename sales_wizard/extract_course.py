import requests

from bs4 import BeautifulSoup

def scrape_page(url):
  response = requests.get(url)
  with open("course.html", 'w') as page:
    page.write(response.text)
    # soup = BeautifulSoup(page, 'html.parser')
#   soup = BeautifulSoup(response.text, 'html.parser')
#   return soup

def extract_course(file_path):
  with open(file_path, 'r') as f:
    html = f.read()

  soup = BeautifulSoup(html, 'html.parser')
  
  with open("courses.md", 'w') as page:
    page.write(soup.get_text())
  

if __name__ == '__main__':
#   url = "https://artifyindiadesign.com/courses/"

#   scrape_page(url)

  extract_course("course.html")
