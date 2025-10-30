import json
import re
import pandas as pd
from pathlib import Path
import glob

def clean_articles(input_file=None, output_file='data/processed/articles.csv'):
    """
    Load articles from JSON, clean text, and save to CSV.
    
    Args:
        input_file: Path to input JSON file (if None, uses most recent file in data/raw/)
        output_file: Path to output CSV file
    
    Returns:
        pandas.DataFrame: Cleaned articles
    """
    # If no input file specified, find the most recent JSON file
    if input_file is None:
        json_files = glob.glob('data/raw/*.json')
        if not json_files:
            raise FileNotFoundError("No JSON files found in data/raw/")
        input_file = max(json_files, key=lambda x: Path(x).stat().st_mtime)
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
    for article in articles:
        # Combine title, description, and content
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        content = article.get('content', '') or ''
        
        combined_text = f"{title} {description} {content}"
        
        # Clean text
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', combined_text)
        # Remove URLs
        clean_text = re.sub(r'http[s]?://\S+|www\.\S+', '', clean_text)
        # Remove multiple spaces
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Only add articles with actual content
        if clean_text:
            processed_articles.append({
                'title': title,
                'text': clean_text,
                'publishedAt': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', '') if isinstance(article.get('source'), dict) else article.get('source', '')
            })
    
    # Create DataFrame
    df = pd.DataFrame(processed_articles)
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Successfully processed {len(df)} articles")
    return df


if __name__ == '__main__':
    df = clean_articles()
    print(f"Processed {len(df)} articles")
    print(df.head())