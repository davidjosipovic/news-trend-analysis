import requests
import json
import os
from datetime import datetime
from pathlib import Path

# Try to load from .env file
try:
    from dotenv import load_dotenv
    # Load from .env file in project root
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv not installed, will use environment variables only
    pass

def load_existing_articles():
    """Load existing articles from all JSON files to avoid duplicates."""
    existing_urls = set()
    raw_dir = Path("data/raw")
    
    if raw_dir.exists():
        for json_file in raw_dir.glob("news_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                    for article in articles:
                        if article.get('url'):
                            existing_urls.add(article['url'])
            except Exception as e:
                print(f"Warning: Could not read {json_file}: {e}")
    
    return existing_urls

def fetch_news(api_key, query="economy", language="en", max_results=50):
    """
    Fetch news articles from NewsData.io API and save to JSON file.
    Only saves NEW articles (deduplicates by URL).
    
    Args:
        api_key (str): NewsData.io API key
        query (str): Search query
        language (str): Language code
        max_results (int): Maximum number of results to fetch (default 50)
        
    Returns:
        list: List of NEW news articles (excluding duplicates)
    """
    # Load existing URLs to avoid duplicates
    existing_urls = load_existing_articles()
    print(f"Found {len(existing_urls)} existing articles in database")
    
    url = "https://newsdata.io/api/1/latest"
    
    params = {
        "apikey": api_key,
        "q": query,
        "language": language
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("status") != "success":
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
        
        # Filter out duplicates
        new_articles = []
        duplicates = 0
        
        for article in data.get("results", [])[:max_results]:
            article_url = article.get("link")
            
            # Skip if already exists
            if article_url in existing_urls:
                duplicates += 1
                continue
            
            new_articles.append({
                "title": article.get("title"),
                "description": article.get("description"),
                "source": article.get("source_name"),
                "publishedAt": article.get("pubDate"),
                "content": article.get("content"),
                "url": article_url
            })
        
        if not new_articles:
            print(f"No new articles found. All {duplicates} fetched articles were duplicates.")
            return []
        
        # Create data/raw directory if it doesn't exist
        os.makedirs("data/raw", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/raw/news_{timestamp}.json"
        
        # Save to JSON file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(new_articles, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully fetched {len(new_articles)} NEW articles (skipped {duplicates} duplicates)")
        print(f"Saved to {filename}")
        return new_articles
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []


if __name__ == "__main__":
    # Get API key from environment variable or .env file
    API_KEY = os.environ.get('NEWS_API_KEY', 'your_api_key_here')
    
    if API_KEY == 'your_api_key_here':
        print("No API key found!")
        print("Please set your NewsData.io API key in one of these ways:")
        print("1. Edit the .env file and add: NEWS_API_KEY=your_actual_key")
        print("2. Or run: export NEWS_API_KEY='your_actual_key'")
        print("Get a free key from: https://newsdata.io/")
        exit(1)
    
    print(f"API key loaded successfully")
    print(f"Fetching news articles from NewsData.io...")
    articles = fetch_news(API_KEY, query="economy", language="en", max_results=50)
