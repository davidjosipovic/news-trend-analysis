import pandas as pd
from transformers import pipeline
import os
import torch

def analyze_sentiment(incremental=True):
    """
    Analyze sentiment of articles using pre-trained RoBERTa model.
    Uses cardiffnlp/twitter-roberta-base-sentiment-latest for 3-way sentiment.
    Applies inference (no training) with batch processing on CPU.
    Saves results with sentiment labels and confidence scores.
    
    Args:
        incremental: If True, only process new articles without sentiment
    """
    # Set CPU threads for optimal performance
    torch.set_num_threads(torch.get_num_threads())
    print(f"Using CPU with {torch.get_num_threads()} threads")
    
    # Initialize sentiment analysis pipeline with CPU optimizations
    print("Loading sentiment analysis model...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",  # 3-way sentiment (pos/neu/neg)
        device=-1,  # Force CPU
        batch_size=16  # Process multiple articles at once
    )
    
    # Load articles (preprocessed, no sentiment yet)
    input_path = "data/processed/articles.csv"
    output_path = "data/processed/articles_with_sentiment.csv"
    
    print(f"Loading articles from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Load existing sentiment analysis if incremental mode
    existing_sentiment = pd.DataFrame()
    if incremental and os.path.exists(output_path):
        print(f"Loading existing sentiment data from {output_path}...")
        existing_sentiment = pd.read_csv(output_path)
        print(f"Found {len(existing_sentiment)} articles with sentiment")
        
        # Identify NEW articles (not in existing sentiment file)
        if len(existing_sentiment) > 0:
            existing_sentiment['_id'] = existing_sentiment['title'].astype(str) + '||' + existing_sentiment['publishedAt'].astype(str)
            df['_id'] = df['title'].astype(str) + '||' + df['publishedAt'].astype(str)
            
            existing_ids = set(existing_sentiment['_id'].values)
            new_articles_mask = ~df['_id'].isin(existing_ids)
            new_articles = df[new_articles_mask].copy()
            
            print(f"Found {len(new_articles)} NEW articles to analyze (out of {len(df)} total)")
            
            if len(new_articles) == 0:
                print("No new articles to process - using existing sentiment data")
                # Clean up temp column
                existing_sentiment = existing_sentiment.drop('_id', axis=1)
                return existing_sentiment
            
            df_to_process = new_articles
        else:
            df_to_process = df.copy()
    else:
        print("No existing sentiment file - processing all articles")
        df_to_process = df.copy()
    
    # Prepare article texts for batch processing (analyze full text, not summaries)
    print(f"Analyzing sentiment on {len(df_to_process)} articles with batch processing...")
    texts = []
    for idx, row in df_to_process.iterrows():
        # Use first 512 characters of text for sentiment (better than waiting for summaries)
        text = row.get('text', '') or ''
        texts.append(text[:512] if text and len(text.strip()) > 0 else "")
    
    # Batch process all summaries at once
    results = sentiment_pipeline(texts, truncation=True, max_length=512)
    
    # Map results to simple categories and save confidence scores
    sentiments = []
    confidences = []
    for result in results:
        label = result['label'].lower()
        if 'pos' in label:
            sentiment = 'positive'
        elif 'neg' in label:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        sentiments.append(sentiment)
        confidences.append(round(result['score'], 4))
    
    print(f"Processed {len(df_to_process)} articles")
    
    # Add sentiment and confidence columns to NEW articles
    df_to_process['sentiment'] = sentiments
    df_to_process['sentiment_confidence'] = confidences
    
    # Merge with existing sentiment data if incremental
    if incremental and not existing_sentiment.empty:
        # Drop temp ID from existing
        if '_id' in existing_sentiment.columns:
            existing_sentiment = existing_sentiment.drop('_id', axis=1)
        if '_id' in df_to_process.columns:
            df_to_process = df_to_process.drop('_id', axis=1)
        
        # Combine old + new
        df_final = pd.concat([existing_sentiment, df_to_process], ignore_index=True)
        print(f"Combined {len(existing_sentiment)} existing + {len(df_to_process)} new = {len(df_final)} total articles")
    else:
        df_final = df_to_process
        if '_id' in df_final.columns:
            df_final = df_final.drop('_id', axis=1)
    
    # Report confidence statistics
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    low_confidence_count = sum(1 for c in confidences if c < 0.7)
    print(f"Average confidence: {avg_confidence:.3f}")
    if low_confidence_count > 0:
        print(f"⚠️  {low_confidence_count} articles have low confidence (< 0.7)")
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    print(f"\nSentiment distribution:\n{df_final['sentiment'].value_counts()}")
    
    return df_final

if __name__ == "__main__":
    analyze_sentiment()