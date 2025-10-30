import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from pathlib import Path
import os

def scrape_article_content(url, timeout=10):
    """
    Scrape full article content from a URL.
    
    Args:
        url (str): Article URL
        timeout (int): Request timeout in seconds
        
    Returns:
        str: Scraped article content or empty string if failed
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Try common article content selectors
        content_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            'main',
            '.story-body',
            '[itemprop="articleBody"]',
        ]
        
        article_text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                # Get text from all paragraphs in the article
                paragraphs = elements[0].find_all('p')
                article_text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                if len(article_text.split()) > 50:  # If we got substantial content
                    break
        
        # Fallback: get all paragraph text if specific selectors didn't work
        if not article_text or len(article_text.split()) < 50:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return article_text
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


def scrape_articles_from_json():
    """
    Load ALL articles from news_*.json files and scrape ONLY those not yet scraped.
    Merges with existing scraped data.
    """
    raw_dir = Path("data/raw")
    
    # Load existing scraped articles to avoid re-scraping
    scraped_file = raw_dir / "articles_scraped.json"
    existing_scraped = {}
    
    if scraped_file.exists():
        try:
            with open(scraped_file, 'r', encoding='utf-8') as f:
                existing_articles = json.load(f)
                for article in existing_articles:
                    existing_scraped[article['url']] = article
            print(f"✓ Found {len(existing_scraped)} already scraped articles")
        except Exception as e:
            print(f"Warning: Could not load existing scraped articles: {e}")
    
    # Collect ALL articles from all news_*.json files
    all_articles = {}  # Use dict to deduplicate by URL
    
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
            print(f"Warning: Could not read {json_file}: {e}")
    
    print(f"
✓ Total unique articles found: {len(all_articles)}")
    
    # Determine which articles need scraping
    to_scrape = []
    for url, article in all_articles.items():
        if url not in existing_scraped:
            to_scrape.append(article)
    
    if not to_scrape:
        print("✓ All articles already scraped! Nothing to do.")
        return existing_scraped
    
    print(f"✓ Need to scrape {len(to_scrape)} NEW articles
")
    
    # Scrape new articles
    scraped_articles = list(existing_scraped.values())  # Start with existing
    successful = 0
    failed = 0
    
    for i, article in enumerate(to_scrape, 1):
        url = article.get('url', '')
        title = article.get('title', 'Unknown')
        
        print(f"[{i}/{len(to_scrape)}] Scraping: {title[:60]}...")
        
        full_text = scrape_article_content(url)
        
        if full_text and len(full_text.split()) > 50:
            successful += 1
            print(f"  ✓ Success! Got {len(full_text.split())} words")
        else:
            failed += 1
            # Use API content as fallback
            full_text = article.get('content', article.get('description', ''))
            print(f"  ⚠️  Failed, using API content ({len(full_text.split())} words)")
        
        # Create scraped article
        scraped_article = {
            'title': article.get('title'),
            'text': full_text,
            'publishedAt': article.get('publishedAt'),
            'source': article.get('source'),
            'url': url
        }
        
        scraped_articles.append(scraped_article)
        
        # Be polite - delay between requests
        if i < len(to_scrape):
            time.sleep(2)
    
    # Save ALL articles (old + new)
    output_file = raw_dir / "articles_scraped.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scraped_articles, f, ensure_ascii=False, indent=2)
    
    print(f"
{'='*70}")
    print(f"✓ SCRAPING COMPLETE")
    print(f"{'='*70}")
    print(f"Total articles in database: {len(scraped_articles)}")
    print(f"Newly scraped: {len(to_scrape)}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    print(f"Saved to: {output_file}")
    
    return scraped_articles


def update_processed_csv_with_scraped_content():
    """
    Update the processed CSV with scraped content instead of API content.
    """
    scraped_file = 'data/raw/articles_scraped.json'
    
    if not os.path.exists(scraped_file):
        print(f"Scraped file not found: {scraped_file}")
        print("Run scraping first!")
        return
    
    print(f"Loading scraped articles from {scraped_file}...")
    with open(scraped_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # Create DataFrame with scraped content
    data = []
    for article in articles:
        # Use scraped content if available, otherwise fall back to API content
        content = article.get('scraped_content') or article.get('content', '')
        
        data.append({
            'title': article.get('title'),
            'text': content,
            'publishedAt': article.get('publishedAt') or article.get('pubDate'),
            'source': article.get('source') or article.get('source_name'),
            'url': article.get('url') or article.get('link'),
            'scraped': article.get('scraped_success', False)
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = 'data/processed/articles.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    scraped_count = df['scraped'].sum()
    print(f"\n✅ Updated {output_file}")
    print(f"   Articles with scraped content: {scraped_count}/{len(df)}")
    print(f"   Average words per article: {df['text'].str.split().str.len().mean():.0f}")
    
    return df


if __name__ == "__main__":
    import sys
    
    # Step 1: Scrape articles
    print("=" * 60)
    print("STEP 1: Scraping full article content from URLs")
    print("=" * 60)
    articles = scrape_articles_from_json(delay=2)  # 2 seconds delay to be polite
    
    if not articles:
        print("No articles to scrape!")
        sys.exit(1)
    
    # Step 2: Update processed CSV
    print("\n" + "=" * 60)
    print("STEP 2: Updating processed CSV with scraped content")
    print("=" * 60)
    df = update_processed_csv_with_scraped_content()
    
    print("\n" + "=" * 60)
    print("✅ ALL DONE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run sentiment analysis: python src/train_sentiment_model.py")
    print("  2. Run topic modeling: python src/train_topic_model.py")
    print("  3. Generate summaries: python src/summarize.py")
