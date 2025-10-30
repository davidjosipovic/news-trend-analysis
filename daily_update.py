#!/usr/bin/env python3
"""
Daily Update Script - Fetches new articles and runs full NLP pipeline
Usage: python daily_update.py [--max-results 50]
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    parser = argparse.ArgumentParser(description='Fetch and process news articles')
    parser.add_argument('--max-results', type=int, default=50, 
                       help='Maximum number of articles to fetch (default: 50)')
    parser.add_argument('--query', type=str, default='economy',
                       help='Search query (default: economy)')
    args = parser.parse_args()
    
    print("="*70)
    print("NEWS TREND ANALYSIS - DAILY UPDATE")
    print("="*70)
    
    # Load environment variables
    env_path = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=env_path)
    API_KEY = os.environ.get('NEWS_API_KEY')
    
    if not API_KEY:
        print("❌ ERROR: NEWS_API_KEY not found in environment!")
        print("   Please set it in .env file")
        sys.exit(1)
    
    # Step 1: Fetch news
    print(f"\n📰 STEP 1: Fetching {args.max_results} articles...")
    print("-"*70)
    from fetch_news import fetch_news
    new_articles = fetch_news(API_KEY, query=args.query, language="en", max_results=args.max_results)
    
    if not new_articles:
        print("\n⚠️  No new articles to process. Database is up to date!")
        return
    
    # Step 2: Scrape full content
    print(f"\n🕷️  STEP 2: Scraping full article content...")
    print("-"*70)
    from scrape_articles import scrape_articles_from_json, update_processed_csv_with_scraped_content
    
    scraped_articles = scrape_articles_from_json()
    update_processed_csv_with_scraped_content()
    
    # Step 3: Sentiment Analysis (preprocessing already done by scrape_articles)
    print(f"\n😊 STEP 3: Running sentiment analysis...")
    print("-"*70)
    from train_sentiment_model import train_sentiment_model
    train_sentiment_model()
    
    # Step 4: Topic Modeling
    print(f"\n📊 STEP 4: Running topic modeling...")
    print("-"*70)
    from train_topic_model import train_topic_model
    train_topic_model()
    
    # Step 5: Summarization
    print(f"\n📝 STEP 5: Generating summaries...")
    print("-"*70)
    from summarize import summarize_articles
    summarize_articles()
    
    print(f"\n{'='*70}")
    print("✅ DAILY UPDATE COMPLETE!")
    print("="*70)
    
    # Show final stats
    import pandas as pd
    final_df = pd.read_csv('data/processed/articles_with_summary.csv')
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total articles: {len(final_df)}")
    
    if 'scraped' in final_df.columns:
        print(f"   Successfully scraped: {final_df['scraped'].sum()}")
    
    if 'sentiment' in final_df.columns:
        print(f"   Sentiment distribution:")
        print(f"      {final_df['sentiment'].value_counts().to_dict()}")
    
    if 'topic' in final_df.columns:
        print(f"   Topic distribution:")
        print(f"      {final_df['topic'].value_counts().to_dict()}")
    
    print(f"\n🎉 Dashboard ready at: http://localhost:8501")
    print(f"   Run: streamlit run dashboard/streamlit_app.py")

if __name__ == "__main__":
    main()
