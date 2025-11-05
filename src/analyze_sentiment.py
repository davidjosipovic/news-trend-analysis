import pandas as pd
from transformers import pipeline, RobertaTokenizerFast
import os
import torch
import sys
import json

def analyze_sentiment(incremental=True, use_adapter=False, adapter_path=None):
    """
    Analyze sentiment of articles using pre-trained RoBERTa model.
    Uses cardiffnlp/twitter-roberta-base-sentiment-latest for 3-way sentiment.
    Optionally loads a fine-tuned adapter for improved performance.
    Applies inference (no training) with batch processing on CPU.
    Saves results with sentiment labels and confidence scores.
    
    Args:
        incremental: If True, only process new articles without sentiment
        use_adapter: If True, use fine-tuned adapter instead of base model
        adapter_path: Path to adapter directory (e.g., './sentiment_adapter_best')
    """
    # CPU setup
    torch.set_num_threads(torch.get_num_threads())
    print(f"Using CPU with {torch.get_num_threads()} threads")
    device = torch.device("cpu")
    pipeline_device = -1
    batch_size = 16
    
    # Initialize sentiment analysis model
    print("Loading sentiment analysis model...")
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Use adapter if specified
    if use_adapter and adapter_path:
        print(f"Loading fine-tuned adapter from: {adapter_path}")
        try:
            from adapters import AutoAdapterModel
            
            # Load model with adapter support
            model = AutoAdapterModel.from_pretrained(model_name)
            tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
            
            # Load adapter
            adapter_name = "sentiment_adapter"
            model.load_adapter(adapter_path, load_as=adapter_name, set_active=True)
            model.set_active_adapters(adapter_name)
            model.to(device)
            model.eval()
            
            print(f"✅ Adapter loaded successfully!")
            print(f"   Active adapters: {model.active_adapters}")
            
            # Load label mapping if available, otherwise use default
            label_mapping_path = os.path.join(adapter_path, "..", "artifacts", "label_mapping.json")
            id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}  # Default
            
            if os.path.exists(label_mapping_path):
                try:
                    with open(label_mapping_path, 'r') as f:
                        label_info = json.load(f)
                        id2label = {int(k): v for k, v in label_info.get("id2label", {}).items()}
                        print(f"   Label mapping: {id2label}")
                except Exception as e:
                    print(f"   ⚠️ Could not load label mapping: {e}")
                    print(f"   Using default: {id2label}")
            else:
                print(f"   Using default label mapping: {id2label}")
            
            # Create custom inference function
            def adapter_pipeline(texts, truncation=True, max_length=512):
                results = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    enc = tokenizer(
                        batch_texts,
                        truncation=truncation,
                        padding=True,
                        return_tensors="pt",
                        max_length=max_length
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    
                    with torch.no_grad():
                        outputs = model(**enc)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=-1)
                        pred_ids = probs.argmax(dim=-1).tolist()
                        scores = probs.max(dim=-1).values.tolist()
                    
                    for pred_id, score in zip(pred_ids, scores):
                        label = id2label.get(pred_id, f"LABEL_{pred_id}")
                        results.append({"label": label, "score": score})
                
                return results
            
            sentiment_pipeline = adapter_pipeline
            using_adapter = True
            
        except ImportError:
            print("⚠️  'adapters' library not installed. Install with: pip install adapter-transformers")
            print("   Falling back to base model...")
            use_adapter = False
            using_adapter = False
        except Exception as e:
            print(f"⚠️  Error loading adapter: {e}")
            print("   Falling back to base model...")
            use_adapter = False
            using_adapter = False
    else:
        using_adapter = False
    
    # Use base model if adapter not used
    if not use_adapter or not using_adapter:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=pipeline_device,
            batch_size=batch_size
        )
        print(f"✅ Base model loaded")
    
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
    import argparse
    parser = argparse.ArgumentParser(description='Sentiment analysis with optional adapter')
    parser.add_argument('--no-adapter', action='store_true', 
                       help='Force use of base model instead of adapter')
    parser.add_argument('--adapter-path', type=str, 
                       default='./models/sentiment_adapter_best',
                       help='Path to adapter directory')
    args = parser.parse_args()
    
    # Auto-detect adapter: use it if exists and not explicitly disabled
    adapter_path = args.adapter_path
    use_adapter = os.path.exists(adapter_path) and not args.no_adapter
    
    if use_adapter:
        print(f"🎯 Using fine-tuned adapter from: {adapter_path}")
        print("   (Use --no-adapter to force base model)")
    
    analyze_sentiment(use_adapter=use_adapter, adapter_path=adapter_path)