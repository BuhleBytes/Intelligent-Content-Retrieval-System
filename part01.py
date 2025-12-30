import requests
import json
import os
import time
import trafilatura
import terminal_ui as ui
from datetime import datetime


os.makedirs("data/raw_trafilatura", exist_ok=True)  # Different folder for comparison

urls = {
    "News": "https://www.bbc.com/future/article/20251218-dian-fossey-the-woman-who-gave-her-life-to-save-the-gorillas",
    "Educational": "https://en.wikipedia.org/wiki/Machine_learning",
    "Technical Documentation": "https://www.tensorflow.org/guide/intro_to_graphs",
    "Research Publication": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4165831/"
}

headers = {
    "User-Agent": "StudentBot/1.0 (University Assignment; mlnhon001@myuct.ac.za)"
}

for category, url in urls.items():
    print(f"Scraping {category} with trafilatura...")
    
    response = requests.get(url, headers=headers)
    
    # Use trafilatura for extraction
    content_text = trafilatura.extract(
        response.text,
        include_formatting=False,
        include_links=False,
        include_images=False,
        include_tables=True,
        include_comments=False
    )
    
    if content_text is None:
        print(f"⚠️ Could not extract content for {category}")
        continue
    
    scraped_data = {
        "url": url,
        "domain": url.split("//")[1].split("/")[0],
        "category": category,
        "timestamp": datetime.now().isoformat(),
        "content": content_text,
        "metadata": {
            "character_count": len(content_text),
            "word_count": len(content_text.split())
        }
    }
    
    filename = category.lower().replace(" ", "_")
    
    with open(f"data/raw_trafilatura/{filename}.json", "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, indent=2, ensure_ascii=False)
    
    print(f"Scraped {len(content_text)} characters from {category}")
    
    time.sleep(2)

print("\n✓ Trafilatura scraping complete!")