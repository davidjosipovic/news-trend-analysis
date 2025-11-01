import pandas as pd
from transformers import pipeline
import os
import torch

def analyze_sentiment():
    """
    Analyze sentiment of articles using pre-trained RoBERTa model.
    Applies inference (no training) with batch processing on CPU.
    Saves results with sentiment labels and confidence scores.
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
    
    print(f"Processed {len(df)} articles")
    
    # Add sentiment and confidence columns
    df['sentiment'] = sentiments
    df['sentiment_confidence'] = confidences
    
    # Report confidence statistics
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    low_confidence_count = sum(1 for c in confidences if c < 0.7)
    print(f"Average confidence: {avg_confidence:.3f}")
    if low_confidence_count > 0:
        print(f"⚠️  {low_confidence_count} articles have low confidence (< 0.7)")
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    print(f"\nSentiment distribution:\n{df['sentiment'].value_counts()}")

if __name__ == "__main__":
    analyze_sentiment()