import pandas as pd
from transformers import pipeline
import os
import torch

def summarize_articles():
    """
    Summarize articles using HuggingFace model and save results.
    CPU-optimized with batch processing.
    """
    # Set CPU threads for optimal performance
    num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    print(f"Using CPU with {num_threads} threads")
    
    # Load the summarization pipeline with CPU optimization
    print("Loading summarization model (optimized for CPU)...")
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1,  # Force CPU
        batch_size=8  # Process multiple articles at once
    )
    
    # Read the input CSV - try with topics first, fall back to sentiment only
    input_path = "data/processed/articles_with_topics.csv"
    if not os.path.exists(input_path):
        input_path = "data/processed/articles_with_sentiment.csv"
    
    print(f"Loading articles from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Prepare texts for batch processing
    print("Generating summaries with batch processing...")
    texts = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        text = row.get('text', '') or row.get('content', '')
        
        # Handle empty or short content
        if pd.isna(text) or len(str(text).split()) < 30:
            continue
        
        # Truncate to avoid model limits (1024 tokens)
        texts.append(str(text)[:2000])
        valid_indices.append(idx)
    
    # Batch process all texts at once (much faster!)
    print(f"Processing {len(texts)} articles...")
    try:
        results = summarizer(
            texts,
            max_length=130,
            min_length=30,
            do_sample=False,
            truncation=True
        )
        
        # Map results back to dataframe
        summaries = [""] * len(df)
        for i, result in enumerate(results):
            summaries[valid_indices[i]] = result['summary_text']
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(texts)} articles")
        
        df['summary'] = summaries
        
    except Exception as e:
        print(f"Batch processing failed: {e}")
        print("Falling back to sequential processing...")
        
        # Fallback: process one by one
        summaries = []
        for idx, row in df.iterrows():
            text = row.get('text', '') or row.get('content', '')
            
            if pd.isna(text) or len(str(text).split()) < 30:
                summaries.append("")
                continue
            
            try:
                text_str = str(text)[:2000]
                summary = summarizer(
                    text_str,
                    max_length=130,
                    min_length=30,
                    do_sample=False
                )
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Error summarizing article {idx}: {e}")
                summaries.append("")
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} articles")
        
        df['summary'] = summaries
    
    # Save to output CSV
    output_path = "data/processed/articles_with_summary.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nSummaries saved to {output_path}")


if __name__ == "__main__":
    summarize_articles()