import json
import re
import pandas as pd
from pathlib import Path
import glob

def clean_articles(input_file='data/raw/articles_scraped.json', output_file='data/processed/articles.csv'):
    """
    Load articles from JSON, clean text, and save to CSV.
    
    Args:
        input_file: Path to input JSON file (defaults to articles_scraped.json with full content)
        output_file: Path to output CSV file
    
    Returns:
        pandas.DataFrame: Cleaned articles
    """
    # Use scraped articles by default (has full text content)
    if input_file == 'data/raw/articles_scraped.json' and not Path(input_file).exists():
        # Fallback to most recent JSON if scraped file doesn't exist
        json_files = glob.glob('data/raw/news_*.json')
        if not json_files:
            raise FileNotFoundError("No JSON files found in data/raw/")
        input_file = max(json_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"Warning: articles_scraped.json not found, using fallback: {input_file}")
    
    print(f"Using input file: {input_file}")
    
    # Load JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list format and dict with 'articles' key
    if isinstance(data, list):
        articles = data
    else:
        articles = data.get('articles', [])
    
    # Process each article
    processed_articles = []
    skipped_paid = 0
    skipped_short = 0
    
    for article in articles:
        # Get title and text
        title = article.get('title', '') or ''
        
        # Try 'text' field first (scraped content), then fallback to description+content (API)
        text = article.get('text', '')
        if not text or text == "ONLY AVAILABLE IN PAID PLANS":
            description = article.get('description', '') or ''
            content = article.get('content', '') or ''
            combined_text = f"{title} {description} {content}"
        else:
            combined_text = text  # Use full scraped text directly
        
        # Skip articles with "ONLY AVAILABLE IN PAID PLANS" or insufficient content
        if "ONLY AVAILABLE IN PAID PLANS" in combined_text:
            skipped_paid += 1
            continue
        
        # Clean text
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', combined_text)
        # Remove URLs
        clean_text = re.sub(r'http[s]?://\S+|www\.\S+', '', clean_text)
        # Remove multiple spaces
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Only add articles with substantial content (minimum 100 words for good summaries)
        word_count = len(clean_text.split())
        if clean_text and word_count >= 100:
            processed_articles.append({
                'title': title,
                'text': clean_text,
                'publishedAt': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', '') if isinstance(article.get('source'), dict) else article.get('source', '')
            })
        else:
            skipped_short += 1
    
    # Create DataFrame
    df = pd.DataFrame(processed_articles)
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    if skipped_paid > 0:
        print(f"Skipped {skipped_paid} articles with restricted/paid content")
    if skipped_short > 0:
        print(f"Skipped {skipped_short} articles with insufficient content (< 100 words)")
    print(f"Successfully processed {len(df)} articles with sufficient content")
    return df


if __name__ == '__main__':
    df = clean_articles()
    print(f"Processed {len(df)} articles")
    print(df.head())