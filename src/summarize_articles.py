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

    # Read the input CSV - always start from the source (topics or sentiment)
    # because articles_with_summary.csv always has all summaries already
    input_path = "data/processed/articles_with_topics.csv"
    if not os.path.exists(input_path):
        input_path = "data/processed/articles_with_sentiment.csv"

    print(f"Loading articles from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Remove duplicates from input
    initial_count = len(df)
    df = df.drop_duplicates(subset=['title', 'publishedAt'], keep='last')
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} duplicate articles from input")
    
    # Check if we have existing summaries to avoid re-processing
    summary_path = "data/processed/articles_with_summary.csv"
    if os.path.exists(summary_path):
        print(f"Loading existing summaries from {summary_path}...")
        df_existing = pd.read_csv(summary_path)
        
        # Identify articles that need summarization (not in existing file)
        # Use title + publishedAt as unique identifier
        if 'title' in df.columns and 'publishedAt' in df.columns:
            # Create unique identifier for each article
            df['_article_id'] = df['title'].astype(str) + '||' + df['publishedAt'].astype(str)
            df_existing['_article_id'] = df_existing['title'].astype(str) + '||' + df_existing['publishedAt'].astype(str)
            
            existing_ids = set(df_existing['_article_id'].values)
            new_articles_mask = ~df['_article_id'].isin(existing_ids)
            new_articles = df[new_articles_mask]
            print(f"Found {len(new_articles)} new articles to summarize (out of {len(df)} total)")
            
            # Merge existing summaries into the source dataframe
            df = df.merge(
                df_existing[['_article_id', 'summary']],
                on='_article_id',
                how='left',
                suffixes=('', '_existing')
            )
            
            # Use existing summary where available
            if 'summary_existing' in df.columns:
                df['summary'] = df['summary_existing'].fillna(df.get('summary', ''))
                df = df.drop('summary_existing', axis=1)
            
            df_to_summarize = new_articles
            
        else:
            print("⚠️  No 'title' or 'publishedAt' column found - processing all articles")
            df_to_summarize = df
    else:
        print("No existing summary file found - processing all articles")
        df_to_summarize = df

    # Prepare texts for batch processing
    print("Generating summaries with batch processing...")
    texts = []
    valid_indices = []

    for idx, row in df_to_summarize.iterrows():
        text = row.get('text', '') or row.get('content', '')

        # Handle empty or short content
        if pd.isna(text) or len(str(text).split()) < 30:
            continue

        # Truncate to avoid model limits (1024 tokens)
        texts.append(str(text)[:2000])
        valid_indices.append(idx)

    # Only process if there are texts to summarize
    if len(texts) > 0:
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

            # Add summaries to the new articles
            summaries_dict = {}
            for i, result in enumerate(results):
                summaries_dict[valid_indices[i]] = result['summary_text']

                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(texts)} articles")
            
            summaries_list = []
            for idx in df_to_summarize.index:
                if idx in summaries_dict:
                    summaries_list.append(summaries_dict[idx])
                else:
                    summaries_list.append('')
            
            df_to_summarize['summary'] = summaries_list
            
            # Update the main dataframe with new summaries
            for idx in df_to_summarize.index:
                if idx in summaries_dict:
                    df.loc[idx, 'summary'] = summaries_dict[idx]

        except Exception as e:
            print(f"Batch processing failed: {e}")
            print("Falling back to sequential processing...")

            # Fallback: process one by one
            summaries = []
            for idx, row in df_to_summarize.iterrows():
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

                if (len(summaries)) % 10 == 0:
                    print(f"Processed {len(summaries)}/{len(df_to_summarize)} articles")

            df_to_summarize['summary'] = summaries
            
            # Update the main dataframe with new summaries
            for idx, summary in zip(df_to_summarize.index, summaries):
                if summary:
                    df.loc[idx, 'summary'] = summary
    else:
        print("No new articles to process - using existing summaries")

    # Clean up temporary ID column if it exists
    if '_article_id' in df.columns:
        df.drop('_article_id', axis=1, inplace=True)
    
    # Remove any duplicate rows before saving
    df = df.drop_duplicates(subset=['title', 'publishedAt'], keep='last')

    # Save to output CSV
    output_path = "data/processed/articles_with_summary.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSummaries saved to {output_path}")

    # Calculate and report quality metrics
    valid_summaries = df[df['summary'].notna() & (df['summary'].str.strip() != '')]
    if len(valid_summaries) > 0:
        summary_word_counts = valid_summaries['summary'].str.split().str.len()
        avg_summary_words = summary_word_counts.mean()
        print(f"\n📊 Summary Quality:")
        print(f"   • Coverage: {len(valid_summaries)}/{len(df)} articles ({len(valid_summaries)/len(df)*100:.1f}%)")
        print(f"   • Average length: {avg_summary_words:.0f} words")
        print(f"   • Range: {summary_word_counts.min()}-{summary_word_counts.max()} words")


if __name__ == "__main__":
    summarize_articles()