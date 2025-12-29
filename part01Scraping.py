import requests
import json
import os
import time
from bs4 import BeautifulSoup
from datetime import datetime

os.makedirs("data/raw", exist_ok=True)
urls = {"News":"https://www.bbc.com/future/article/20251218-dian-fossey-the-woman-who-gave-her-life-to-save-the-gorillas", "Educational":"https://en.wikipedia.org/wiki/Machine_learning", "Technical Documentation":"https://www.tensorflow.org/guide/intro_to_graphs", "Reasearch Publication":"https://pmc.ncbi.nlm.nih.gov/articles/PMC4165831/"}
headers = {
     "User-Agent": "StudentBot/1.0 (University Assignment; mlnhon001@myuct.ac.za)"
 }

for category, url in urls.items():
    response =  requests.get(url, headers=headers)
    parsed_response = BeautifulSoup(response.text, "html.parser")
    content = None
    if category == "News":
        content = parsed_response.find("article")
    elif category == "Educational":
        content = parsed_response.find("main", id="content", class_="mw-body")
    elif category == "Technical Documentation":
        content = parsed_response.find("article", class_="devsite-article")
    else:
        content = parsed_response.find("main", id="main-content")

    content_text = content.get_text(separator="\n\n", strip=True)

    scraped_data = {
        "url":url,
        "domain": url.split("//")[1].split("/")[0],
        "category":category,
        "timestamp":datetime.now().isoformat(),
        "content": content_text,
        "metadata":{
            "character_count": len(content_text),
            "word_count": len(content_text.split())
        }
    }

    filename = category.lower().replace(" ", "_")

    with open(f"data/raw/{filename}.json", "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, indent=2, ensure_ascii=False)
    
    print(f"Scraped {len(content_text)} characters from {url}")

    time.sleep(2)
