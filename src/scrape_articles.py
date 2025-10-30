import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from pathlib import Path
import os

def scrape_article_content(url, timeout=10):
    """Scrape full article content from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Try common article content selectors
        content_selectors = [
            'article', '.article-content', '.post-content', '.entry-content',
            '.content', 'main', '.story-body', '[itemprop="articleBody"]'
        ]
        
        article_text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                paragraphs = elements[0].find_all('p')
                article_text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                if len(article_text.split()) > 50:
                    break
        
        # Fallback
        if not article_text or len(article_text.split()) < 50:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return article_text
        
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")
        return ""

def scrape_articles_from_json():
    """Load ALL articles and scrape only new ones."""
    raw_dir = Path("data/raw")
    
    # Load existing scraped articles
    scraped_file = raw_dir / "articles_scraped.json"
    existing_scraped = {}
    
    if scraped_file.exists():
        try:
            with open(scraped_file, 'r', encoding='utf-8') as f:
                existing_articles = json.load(f)
                for article in existing_articles:
                    existing_scraped[article['url']] = article
            print(f"Found {len(existing_scraped)} already scraped articles")
        except Exception as e:
            print(f"Warning: {e}")
    
    # Collect ALL articles from news_*.json files
    all_articles = {}
    
    for json_file in sorted(raw_dir.glob("news_*.json")):
        print(f"Reading {json_file.name}...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
                for article in articles:
                    url = article.get('url')
                    if url and url not in all_articles:
                        all_articles[url] = article
        except Exception as e:
            print(f"Warning: {e}")
    
    print(f"Total unique articles: {len(all_articles)}")
    
    # Determine which articles need scraping
    to_scrape = []
    for url, article in all_articles.items():
        if url not in existing_scraped:
            to_scrape.append(article)
    
    if not to_scrape:
        print("All articles already scraped!")
        return list(existing_scraped.values())
    
    print(f"Need to scrape {len(to_scrape)} NEW articles\n")
    
    # Scrape new articles
    scraped_articles = list(existing_scraped.values())
    successful = 0
    failed = 0
    
    for i, article in enumerate(to_scrape, 1):
        url = article.get('url', '')
        title = article.get('title', 'Unknown')
        
        print(f"[{i}/{len(to_scrape)}] {title[:60]}...")
        
        full_text = scrape_article_content(url)
        
        if full_text and len(full_text.split()) > 50:
            successful += 1
            print(f"  Success! {len(full_text.split())} words")
        else:
            failed += 1
            full_text = article.get('content', article.get('description', ''))
            print(f"  Failed, using API content")
        
        scraped_article = {
            'title': article.get('title'),
            'text': full_text,
            'publishedAt': article.get('publishedAt'),
            'source': article.get('source'),
            'url': url
        }
        
        scraped_articles.append(scraped_article)
        
        if i < len(to_scrape):
            time.sleep(2)
    
    # Save ALL articles
    output_file = raw_dir / "articles_scraped.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scraped_articles, f, ensure_ascii=False, indent=2)
    
    print(f"\nSCRAPING COMPLETE")
    print(f"Total articles: {len(scraped_articles)}")
    print(f"Newly scraped: {len(to_scrape)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    return scraped_articles

def update_processed_csv_with_scraped_content():
    """Update CSV with scraped content."""
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    scraped_file = raw_dir / "articles_scraped.json"
    
    if not scraped_file.exists():
        print(f"Error: {scraped_file} not found")
        return
    
    with open(scraped_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    df = pd.DataFrame(articles)
    df['scraped'] = df['text'].str.len() > 100
    
    output_file = processed_dir / "articles.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Processed {len(df)} articles")
    print(f"Scraped successfully: {df['scraped'].sum()}/{len(df)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    print("SCRAPING NEWS ARTICLES")
    scraped_articles = scrape_articles_from_json()
    
    if scraped_articles:
        print("\nUPDATING PROCESSED DATA")
        update_processed_csv_with_scraped_content()
        print("\nAll done!")
