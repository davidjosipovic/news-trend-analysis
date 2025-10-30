import pandas as pd
from transformers import pipeline
import os
import torch

def train_sentiment_model():
    """
    Load articles from CSV, analyze sentiment using HuggingFace model,
    and save results with sentiment column.
    Optimized for CPU with batch processing.
    """
    # Set CPU threads for optimal performance
    torch.set_num_threads(torch.get_num_threads())
    print(f"Using CPU with {torch.get_num_threads()} threads")
    
    # Initialize sentiment analysis pipeline with CPU optimizations
    print("Loading sentiment analysis model...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1,  # Force CPU
        batch_size=16  # Process multiple articles at once
    )
    
    # Load articles
    input_path = "data/processed/articles.csv"
    output_path = "data/processed/articles_with_sentiment.csv"
    
    print(f"Loading articles from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Prepare texts for batch processing
    print("Analyzing sentiment with batch processing...")
    texts = []
    for idx, row in df.iterrows():
        text = row.get('content', '') or row.get('text', '') or row.get('title', '')
        # Truncate text to avoid model limits (514 tokens)
        texts.append(text[:2000] if text and len(text.strip()) > 0 else "")
    
    # Batch process all texts at once (much faster!)
    results = sentiment_pipeline(texts, truncation=True, max_length=512)
    
    # Map results to simple categories
    sentiments = []
    for result in results:
        label = result['label'].lower()
        if 'pos' in label:
            sentiment = 'positive'
        elif 'neg' in label:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        sentiments.append(sentiment)
    
    print(f"Processed {len(df)} articles")
    
    # Add sentiment column
    df['sentiment'] = sentiments
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    print(f"\nSentiment distribution:\n{df['sentiment'].value_counts()}")

if __name__ == "__main__":
    train_sentiment_model()