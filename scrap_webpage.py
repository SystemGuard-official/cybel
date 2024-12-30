import os
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import json
from pathlib import Path
from collections import defaultdict
import re
import unicodedata

# Clean and normalize text
def clean_text(text):
    return unicodedata.normalize("NFKD", text).strip()

# Filter out irrelevant tags
def is_relevant_tag(tag):
    irrelevant_classes = ['nav', 'footer', 'header', 'sidebar']
    return not any(cls in tag.get('class', '') for cls in irrelevant_classes)

# Extract schema.org metadata
def extract_schema_metadata(soup):
    schema_metadata = {}
    for meta in soup.find_all('script', type='application/ld+json'):
        try:
            metadata = json.loads(meta.string)
            schema_metadata.update(metadata)
        except Exception:
            continue
    return schema_metadata

# Sanitize filenames for saving
def sanitize_filename(url):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', url)

# Async function to scrape a webpage
async def scrape_page(session, url):
    try:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"Failed to fetch the page {url}: {response.status}")
                return None

            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')

            # Extract metadata
            title_tag = soup.find('title')
            title = clean_text(title_tag.text) if title_tag else ""
            # schema_metadata = extract_schema_metadata(soup)

            metadata = {
                "title": title,
                "source": url,
            }

            # Extract and organize content hierarchically
            body_content = []
            hierarchy = defaultdict(list)
            current_header = None

            for element in soup.find_all(['h1', 'h2', 'h3', 'p']):
                if element.name.startswith('h'):
                    current_header = clean_text(element.text)
                    hierarchy[current_header] = []
                elif element.name == 'p' and current_header:
                    hierarchy[current_header].append(clean_text(element.text))

            # Convert hierarchy to markdown
            for header, paragraphs in hierarchy.items():
                body_content.append(f"# {header}")
                body_content.extend(paragraphs)

            # Extract tables
            for table in soup.find_all('table'):
                table_rows = []
                headers = [clean_text(header.text) for header in table.find_all('th')]
                if headers:
                    table_rows.append(' | '.join(headers))
                    table_rows.append(' | '.join(['---'] * len(headers)))
                for row in table.find_all('tr'):
                    cols = [clean_text(td.text) for td in row.find_all('td')]
                    if cols:
                        table_rows.append(' | '.join(cols))
                if table_rows:
                    body_content.append('\n'.join(table_rows))

            # Extract links
            for link in soup.find_all('a', href=True):
                link_text = clean_text(link.text)
                link_url = link['href']
                body_content.append(f"[{link_text}]({link_url})")

            # Combine content
            markdown = "\n\n".join(body_content)

            result = {
                "context": markdown,
                "metadata": metadata
            }

            return {"url": url, "data": result}

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# Async function to process all URLs from a file
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

# Entry point
if __name__ == "__main__":
    input_file = "urls.txt"  # File containing URLs separated by newlines
    output_dir = "src/input_data"  # Directory to save JSON files
    os.makedirs(output_dir, exist_ok=True)

    asyncio.run(process_urls(input_file, output_dir))
