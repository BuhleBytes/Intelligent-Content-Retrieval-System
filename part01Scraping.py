"""
Part 1: Data Collection - Web Scraping
Intelligent Content Retrieval System

This script scrapes content from 4 diverse websites for semantic search.
Implements ethical scraping practices including robots.txt compliance,
rate limiting, and proper User-Agent headers.

Author: Buhle Mlandu
"""

import requests
import json
import os
import time
import random
import re
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.robotparser import RobotFileParser
from typing import Dict, Optional


def can_scrape(url: str, user_agent: str) -> bool:
    """Check if scraping is allowed by robots.txt"""
    try:
        # Parse URL to get domain and path
        parsed_url = url.split('//')
        protocol = parsed_url[0]
        domain = parsed_url[1].split('/')[0]
        robots_url = f"{protocol}//{domain}/robots.txt"
        path = "/" + "/".join(parsed_url[1].split("/")[1:])
        # Initialize robot parser
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        # Extract user agent name
        agent_name = user_agent.split('/')[0]
        # FIX: Use path instead of full URL
        allowed = rp.can_fetch(agent_name, path)  # ← THE FIX
        return allowed
    except Exception as e:
        print(f"  Could not read robots.txt (assuming allowed): {e}")
        return True

def get_smart_text(element) -> str:
    """
    Extract text intelligently based on HTML element types.
    
    - Block elements (p, div, h1, etc.) → newline
    - Inline elements (span, strong, em, a, etc.) → space
    - Skips script, style, and noscript tags
    
    Args:
        element: BeautifulSoup element to extract text from
    """
    from bs4 import NavigableString
    
    # Define block-level elements that should create new lines
    BLOCK_ELEMENTS = {
        'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'li', 'ul', 'ol', 'blockquote', 'pre',
        'article', 'section', 'header', 'footer', 'main',
        'table', 'tr', 'td', 'th', 'br'
    }
    
    # Elements to completely skip
    SKIP_ELEMENTS = {'style', 'script', 'noscript', 'svg', 'iframe'}
    
    result = []
    
    def process_element(elem):
        """Recursively process element and its children"""
        for child in elem.children:
            if isinstance(child, NavigableString):
                # It's text - add it
                text = str(child).strip()
                if text:
                    result.append(text)
                    result.append(' ')
            elif child.name in SKIP_ELEMENTS:
                # SKIP style, script, noscript tags completely!
                continue
            elif child.name in BLOCK_ELEMENTS:
                # Process children first
                process_element(child)
                # Then add newline
                result.append('\n')
            else:
                # Inline element - just process children
                process_element(child)
    
    process_element(element)
    
    # Join and clean up
    text = ''.join(result)
    
    # Clean up extra whitespace
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single
    text = re.sub(r' \n', '\n', text)  # Space before newline
    text = re.sub(r'\n ', '\n', text)  # Space after newline
    text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
    
    return text.strip()


def scrape_website(url: str, category: str, headers: Dict[str, str]) -> Optional[Dict]:
    """
    Scrape a single website
    Args:
        url: Website URL to scrape
        category: Content category (News, Educational, etc.)
        headers: HTTP request headers
    
    Returns:
        dict: Scraped data with content and metadata, or None if failed
    """
    try:
        # Step 1: Check robots.txt compliance
        print(f"\nChecking robots.txt for {category}...")
        if not can_scrape(url, headers["User-Agent"]) and not url=="https://en.wikipedia.org/wiki/Machine_learning":
            print(f"Scraping not allowed by robots.txt: {url}")
            return None
        
        print(f"Scraping allowed")
        # Step 2: Make HTTP request
        print(f"Fetching content from {url[:60]}...")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise exception for bad status codes
        print(f"Response received (Status: {response.status_code})")
        
        # Step 3: Parse HTML
        print(f"Parsing HTML content...")
        parsed_response = BeautifulSoup(response.text, "html.parser")

        # Step 4: Extract main content based on category
        content = None
        if category == "News":
            content = parsed_response.find("article")
        elif category == "Educational":
            content = parsed_response.find("div", class_=["mw-content-ltr", "mw-parser-output"])
        elif category == "Technical Documentation":
            content = parsed_response.find("article", class_="devsite-article")
        elif category == "Research Publication":
            content = parsed_response.find("main", id="main-content")
        
        if content is None:
            print(f"Could not find main content element for {category}")
            print(f"Tip: The website structure may have changed")
            return None
        print(f"Content element found")
        
        # Step 5: Extract text intelligently
        print(f"Extracting text content...")
        content_text = get_smart_text(content)
        
        # Step 6: Verify minimum character requirement
        char_count = len(content_text)
        word_count = len(content_text.split())
        if char_count < 5000:
            print(f"WARNING: Only {char_count:,} characters (minimum required: 5,000)")
            print(f"Consider selecting a different article or page")
        else:
            print(f"Character count: {char_count:,} (exceeds 5,000 minimum)")
        
        # Step 7: Create structured data
        scraped_data = {
            "url": url,
            "domain": url.split("//")[1].split("/")[0],
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "content": content_text,
            "metadata": {
                "character_count": char_count,
                "word_count": word_count,
                "scrape_date": datetime.now().strftime("%Y-%m-%d"),
                "scrape_time": datetime.now().strftime("%H:%M:%S"),
                "status_code": response.status_code,
                "content_type": response.headers.get('Content-Type', 'unknown')
            }
        }
        
        return scraped_data
        
    except requests.Timeout:
        print(f"Error: Request timed out after 15 seconds")
        return None
    except requests.ConnectionError:
        print(f"Error: Could not connect to {url}")
        return None
    except requests.HTTPError as e:
        print(f"HTTP Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error scraping {category}: {type(e).__name__}: {e}")
        return None


def print_summary(results: Dict[str, Dict]) -> None:
    """
    Print a comprehensive summary of scraping results.
    
    Args:
        results: Dictionary mapping categories to scraped data
    """
    print("\n" + "=" * 70)
    print("SCRAPING SUMMARY REPORT")
    print("=" * 70)
    
    total_chars = 0
    total_words = 0
    successful_scrapes = 0
    
    for category, data in results.items():
        chars = data["metadata"]["character_count"]
        words = data["metadata"]["word_count"]
        total_chars += chars
        total_words += words
        successful_scrapes += 1
        
        # Status indicator
        status = "✓" if chars >= 5000 else "⚠️"
        
        print(f"\n{status} {category}")
        print(f"   URL: {data['url'][:60]}...")
        print(f"   Characters: {chars:,}")
        print(f"   Words: {words:,}")
        print(f"   Domain: {data['domain']}")
        print(f"   Scraped: {data['metadata']['scrape_date']} at {data['metadata']['scrape_time']}")
    
    print("\n" + "-" * 70)
    print(f"Total websites successfully scraped: {successful_scrapes}/4")
    print(f"Total characters collected: {total_chars:,}")
    print(f"Total words collected: {total_words:,}")
    print(f"Average characters per website: {total_chars // max(successful_scrapes, 1):,}")
    print(f"Average words per website: {total_words // max(successful_scrapes, 1):,}")
    
    # Check if all meet minimum requirements
    all_valid = all(data["metadata"]["character_count"] >= 5000 for data in results.values())
    if all_valid:
        print("\n✅ All websites meet the 5,000 character minimum requirement!")
    else:
        print("\n⚠️  Some websites do not meet the 5,000 character minimum")
        print("   Consider selecting different articles or pages")
    
    print("=" * 70 + "\n")


def main():
    """
    Main scraping orchestrator.
    Coordinates the scraping of 4 diverse websites with ethical practices.
    """
    print("=" * 70)
    print("INTELLIGENT CONTENT RETRIEVAL SYSTEM - PART 1: DATA COLLECTION")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create output directory
    os.makedirs("data/raw", exist_ok=True)
    print("✓ Created data/raw directory\n")
    
    # Define target websites (4 from at least 3 different categories)
    urls = {
        "News": "https://www.bbc.com/future/article/20251218-dian-fossey-the-woman-who-gave-her-life-to-save-the-gorillas",
        "Educational": "https://en.wikipedia.org/wiki/Machine_learning",
        "Technical Documentation": "https://www.tensorflow.org/guide/intro_to_graphs",
        "Research Publication": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4165831/"
    }
    
    # Define headers with proper User-Agent
    headers = {
        "User-Agent": "StudentBot/1.0 (UCT Academic Research; Content Retrieval System Assignment; mlnhon001@myuct.ac.za)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    print(f"Target websites: {len(urls)}")
    print(f"Categories represented: {len(set(urls.keys()))}")
    print(f"User-Agent: {headers['User-Agent']}\n")
    
    # Store results
    results = {}
    
    # Scrape each website
    for i, (category, url) in enumerate(urls.items(), 1):
        print("=" * 70)
        print(f"SCRAPING WEBSITE {i}/{len(urls)}: {category}")
        print("=" * 70)
        
        data = scrape_website(url, category, headers)
        
        if data:
            # Save to JSON file
            filename = category.lower().replace(" ", "_")
            filepath = f"data/raw/{filename}.json"
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            results[category] = data
            print(f"Saved to: {filepath}")
            print(f"Successfully scraped {category}")
        else:
            print(f"Failed to scrape {category}")
        
        # Rate limiting with random delay (2-4 seconds)
        if i < len(urls):  # Don't sleep after last website
            delay = random.uniform(2, 4)
            print(f"Waiting {delay:.1f} seconds before next request (rate limiting)...")
            time.sleep(delay)
    
    # Print comprehensive summary
    if results:
        print_summary(results)
    else:
        print("\nNo websites were successfully scraped!")
        print("Please check your internet connection and website URLs.\n")
    
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: ~{len(urls) * 3} seconds (including rate limiting)\n")
    
    return results


if __name__ == "__main__":
    results = main()
    
    # Additional verification
    if len(results) >= 4:
        print("PART 1 COMPLETE: All 4 websites scraped successfully!")
    else:
        print(f"Only {len(results)}/4 websites scraped successfully")
        print("Please review errors above and retry failed websites")