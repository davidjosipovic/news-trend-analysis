#!/usr/bin/env python
"""
Example script showing how to use the news trend analysis pipeline.
This demonstrates the complete workflow from fetching news to visualization.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fetch_news import fetch_news
from preprocess import clean_articles
from train_sentiment_model import train_sentiment_model
from summarize import summarize_articles

def main():
    """Run the complete pipeline."""
    
    print("=" * 60)
    print("News Trend Analysis Pipeline")
    print("=" * 60)
    
    # Step 1: Fetch news articles
    print("\n[1/4] Fetching news articles...")
    print("-" * 60)
    
    # You need to get an API key from https://newsapi.org/
    api_key = os.environ.get('NEWS_API_KEY', 'your_api_key_here')
    
    if api_key == 'your_api_key_here':
        print("⚠️  Warning: Using placeholder API key.")
        print("   Get a free API key from: https://newsapi.org/")
        print("   Then set it: export NEWS_API_KEY='your_actual_key'")
        print("\n   Skipping fetch step. Using existing data if available...")
    else:
        articles = fetch_news(api_key, query="technology OR economy", language="en")
        if not articles:
            print("❌ Failed to fetch articles. Check your API key.")
            return
    
    # Step 2: Preprocess articles
    print("\n[2/4] Preprocessing articles...")
    print("-" * 60)
    try:
        df = clean_articles()
        print(f"✅ Successfully preprocessed {len(df)} articles")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please run step 1 first to fetch articles.")
        return
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return
    
    # Step 3: Sentiment analysis
    print("\n[3/4] Analyzing sentiment...")
    print("-" * 60)
    try:
        train_sentiment_model()
        print("✅ Sentiment analysis complete")
    except Exception as e:
        print(f"❌ Error during sentiment analysis: {e}")
        return
    
    # Step 4: Summarize articles
    print("\n[4/4] Generating summaries...")
    print("-" * 60)
    try:
        summarize_articles()
        print("✅ Summaries generated successfully")
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        return
    
    # Done!
    print("\n" + "=" * 60)
    print("✅ Pipeline complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the dashboard: streamlit run dashboard/streamlit_app.py")
    print("2. View results in: data/processed/articles_with_summary.csv")
    print("3. For topic modeling: python src/train_topic_model.py")
    print("4. For MLflow tracking: python src/evaluate.py")
    print()

if __name__ == "__main__":
    main()
