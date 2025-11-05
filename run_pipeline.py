#!/usr/bin/env python
"""
Complete news trend analysis pipeline.
Runs all steps from fetching articles to generating summaries and dashboard-ready data.
"""

import os
import sys
import subprocess

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fetch_articles import fetch_news
from preprocess_articles import clean_articles
from analyze_sentiment import analyze_sentiment
from discover_topics import discover_topics
from summarize_articles import summarize_articles
from evaluate_pipeline import evaluate_pipeline

def main():
    """Run the complete 7-step pipeline with evaluation."""
    
    print("=" * 60)
    print("📰 News Trend Analysis Pipeline")
    print("=" * 60)
    
    # Step 1: Fetch news articles
    print("\n[1/7] 🔍 Fetching latest news articles from NewsData.io (all topics)...")
    print("-" * 60)
    
    api_key = os.environ.get('NEWS_API_KEY', 'your_api_key_here')
    
    if api_key == 'your_api_key_here':
        print("⚠️  Warning: No API key found.")
        print("   Get a free API key from: https://newsdata.io/")
        print("   Then create .env file: NEWS_API_KEY=your_actual_key")
        print("\n   Skipping fetch step. Using existing data if available...")
    else:
        try:
            # query=None fetches latest across all topics
            articles = fetch_news(api_key, query=None, language="en", max_results=50)
            if not articles:
                print("❌ No articles fetched. Check your API key or query.")
                return
            print(f"✅ Fetched {len(articles)} articles")
        except Exception as e:
            print(f"❌ Error fetching articles: {e}")
            return
    
    # Step 2: Scrape full article content
    print("\n[2/7] 🕷️  Scraping full article content from websites...")
    print("-" * 60)
    try:
        result = subprocess.run(['python3', 'src/scrape_articles.py'], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Articles scraped successfully")
        else:
            print(f"⚠️  Scraping had some issues: {result.stderr}")
            print("   Continuing with available data...")
    except subprocess.TimeoutExpired:
        print("⚠️  Scraping timeout. Continuing with partial data...")
    except Exception as e:
        print(f"⚠️  Scraping error: {e}. Continuing with API data...")
    
    # Step 3: Preprocess and clean articles
    print("\n[3/7] 🧹 Preprocessing and cleaning articles...")
    print("-" * 60)
    try:
        df = clean_articles()
        print(f"✅ Successfully preprocessed {len(df)} articles (200+ words, no paid content)")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please run steps 1-2 first to fetch and scrape articles.")
        return
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return
    
    # Step 4: Sentiment analysis
    print("\n[4/7] 😊 Analyzing sentiment...")
    print("-" * 60)
    try:
        # Auto-detect and use adapter if available
        adapter_path = 'models/sentiment_adapter_best'
        if os.path.exists(adapter_path):
            print(f"🎯 Using fine-tuned adapter from: {adapter_path}")
            analyze_sentiment(use_adapter=True, adapter_path=adapter_path)
        else:
            print("📊 Using base model (no adapter found)")
            analyze_sentiment()
        print("✅ Sentiment analysis complete")
    except Exception as e:
        print(f"❌ Error during sentiment analysis: {e}")
        return
    
    # Step 5: Topic modeling
    print("\n[5/7] 🏷️  Discovering topics with BERTopic...")
    print("-" * 60)
    try:
        discover_topics()
        print("✅ Topic modeling complete")
    except Exception as e:
        print(f"❌ Error during topic modeling: {e}")
        return
    
    # Step 6: Generate summaries
    print("\n[6/7] 📝 Generating summaries with DistilBART...")
    print("-" * 60)
    try:
        summarize_articles()
        print("✅ Summaries generated successfully")
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        return
    
    # Step 7: Evaluate pipeline quality
    print("\n[7/7] 📊 Evaluating pipeline quality...")
    print("-" * 60)
    try:
        evaluation_results = evaluate_pipeline()
        print("✅ Evaluation complete")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("   Pipeline completed but evaluation failed.")
    
    # Done!
    print("\n" + "=" * 60)
    print("🎉 Pipeline Complete!")
    print("=" * 60)
    print("\n📊 Output files created:")
    print("   • data/processed/articles.csv")
    print("   • data/processed/articles_with_sentiment.csv")
    print("   • data/processed/articles_with_topics.csv")
    print("   • data/processed/articles_with_summary.csv")
    print("   • data/evaluation/evaluation_report.json")
    print("   • data/evaluation/evaluation_report.txt")
    print("\n🚀 Next steps:")
    print("   1. View dashboard: streamlit run dashboard/streamlit_app.py")
    print("   2. Access at: http://localhost:8501")
    print()

if __name__ == "__main__":
    main()
