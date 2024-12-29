import os
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import json
from pathlib import Path

# Function to scrape the webpage asynchronously
async def scrape_page(session, url):
    try:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"Failed to fetch the page {url}: {response.status}")
                return None

            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')

            # Extract metadata
            title = soup.find('title').text if soup.find('title') else ""
            description = soup.find('meta', {'name': 'description'})
            description = description['content'] if description else ""
            language = soup.find('html')['lang'] if soup.find('html') and 'lang' in soup.find('html').attrs else ""
            keywords = soup.find('meta', {'name': 'keywords'})
            keywords = keywords['content'] if keywords else ""

            metadata = {
                "title": title,
                "url": url,
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

            # convert table in some readable format 
            for table in soup.find_all('table'):
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    cols = [ele.text.strip() for ele in cols]
                    body_content.append(' | '.join(cols))
            

            # Append image alt and src | not needed for now
            # for img in soup.find_all('img'):
            #     alt = img.get('alt', "")
            #     src = img.get('src', "")
            #     if alt or src:
            #         body_content.append(f"![{alt}]({src})")

            # Join the body content with newlines
            markdown = "\n\n".join(body_content)

            result = {
                "context": markdown,
                "metadata": metadata
            }

            return {"url": url, "data": result}

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# Function to process all URLs from a file
async def process_urls(input_file, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        tasks = []

        with open(input_file, 'r') as file:
            urls = [line.strip() for line in file if line.strip()]

        for url in urls:
            tasks.append(scrape_page(session, url))

        results = await asyncio.gather(*tasks)

        # Save results to JSON files
        for result in results:
            if result:
                url = result['url']
                data = result['data']
                filename = f"{output_path / sanitize_filename(url)}.json"
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4)

# Helper function to sanitize filenames
import re
def sanitize_filename(url):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', url)

# Entry point
if __name__ == "__main__":
    input_file = "urls.txt"  # File containing URLs separated by newlines
    output_dir = "src/input_data"  # Directory to save JSON files
    os.makedirs(output_dir, exist_ok=True)


    asyncio.run(process_urls(input_file, output_dir))
