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

# Sanitize filenames for saving
def sanitize_filename(url):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', url)

# Convert hierarchy to markdown with clear markers
def hierarchy_to_markdown_with_markers(hierarchy):
    markdown = []
    for header, paragraphs in hierarchy.items():
        markdown.append(f"<header>{header}</header>")  # Mark the header
        for paragraph in paragraphs:
            markdown.append(f"<para>{paragraph}</para>")  # Mark each paragraph
    return "\n".join(markdown)

# Extract tables in markdown with markers
def extract_tables_with_markers(soup):
    tables_markdown = []
    for table in soup.find_all('table'):
        table_rows = []
        headers = [clean_text(header.text) for header in table.find_all('th')]
        if headers:
            table_rows.append(f"<table_header>{' | '.join(headers)}</table_header>")
            table_rows.append(f"<table_divider>{' | '.join(['---'] * len(headers))}</table_divider>")
        for row in table.find_all('tr'):
            cols = [clean_text(td.text) for td in row.find_all('td')]
            if cols:
                table_rows.append(f"<table_row>{' | '.join(cols)}</table_row>")
        if table_rows:
            tables_markdown.append("<table_start>")
            tables_markdown.extend(table_rows)
            tables_markdown.append("<table_end>")
    return "\n".join(tables_markdown)

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

            metadata = {
                "title": title,
                "source": url,
            }

            # Extract and organize content hierarchically
            hierarchy = defaultdict(list)
            current_header = None

            for element in soup.find_all(['h1', 'h2', 'h3', 'p']):
                if element.name.startswith('h'):
                    current_header = clean_text(element.text)
                    hierarchy[current_header] = []
                elif element.name == 'p' and current_header:
                    hierarchy[current_header].append(clean_text(element.text))

            # Generate markdown with markers
            markdown_hierarchy = hierarchy_to_markdown_with_markers(hierarchy)

            # Extract tables
            tables_markdown = extract_tables_with_markers(soup)

            # Combine content
            body_content = [markdown_hierarchy]
            if tables_markdown:
                body_content.append("<section>Tables</section>")
                body_content.append(tables_markdown)

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
