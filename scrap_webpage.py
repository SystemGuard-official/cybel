import requests
from bs4 import BeautifulSoup
import json

from sympy import im

# Function to scrape the webpage
def scrape_page(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the page: {response.status_code}")

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract metadata
    title = soup.find('title').text if soup.find('title') else ""
    description = soup.find('meta', {'name': 'description'})
    description = description['content'] if description else ""
    language = soup.find('html')['lang'] if soup.find('html') and 'lang' in soup.find('html').attrs else ""
    keywords = soup.find('meta', {'name': 'keywords'})
    keywords = keywords['content'] if keywords else ""

    metadata = {
        "title": title,
        "description": description,
        "language": language,
        "keywords": keywords,
    }

    # Extract markdown-like content
    body_content = []

    # Append header text with level
    for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        level = header.name
        body_content.append(f"{'#' * int(level[1])} {header.text.strip()}")

    # Append paragraph text
    for paragraph in soup.find_all('p'):
        body_content.append(paragraph.text.strip())

    # Append image alt and src
    for img in soup.find_all('img'):
        alt = img.get('alt', "")
        src = img.get('src', "")
        if alt or src:
            body_content.append(f"![{alt}]({src})")

    # Join the body content with newlines
    markdown = "\n\n".join(body_content)

    result = {
        "markdown": markdown,
        "metadata": metadata
    }

    return result

# Example usage
url = "https://walkinthewild.co.in/discover-the-top-5-migratory-birds-that-visit-india-every-year/"
data = scrape_page(url)

# Print the result as formatted JSON
output_dir = "src/input_data"
import string
random_name = ''.join([string.ascii_letters[(ord(c) - 65) % 26] for c in url])
output_file = f"{output_dir}/{random_name}.json"
with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)
