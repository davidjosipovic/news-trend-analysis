import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from datetime import datetime


def detect_duplicates_and_spread(df: pd.DataFrame, similarity_threshold: float = 0.70) -> pd.DataFrame:
    """
    Detect duplicate/similar articles and track news spread across sources.
    
    Identifies:
    1. Exact duplicates (same title, different source)
    2. Semantic duplicates (different title, similar content)
    
    Args:
        df: DataFrame with 'title', 'text', 'source', 'publishedAt' columns
        similarity_threshold: Minimum similarity to consider as duplicate (0.70 = 70%)
    
    Returns:
        DataFrame with added columns: is_original, duplicate_of, similarity_score, 
        original_source, similarity_category, spread_count
    """
    print("🔍 Detecting duplicates and tracking news spread...")
    
    # Initialize columns
    df = df.copy()
    df['is_original'] = True
    df['duplicate_of'] = None
    df['similarity_score'] = 1.0
    df['original_source'] = df['source']
    df['similarity_category'] = 'original'
    
    # Parse dates for chronological ordering
    df['parsed_date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    
    # Step 1: Group by exact title match
    print("  Step 1: Finding exact title matches...")
    title_groups = df.groupby('title').groups
    
    exact_duplicates = 0
    for title, indices in title_groups.items():
        if len(indices) > 1:
            # Sort by date - earliest is original
            group = df.loc[indices].sort_values('parsed_date')
            original_idx = group.index[0]
            original_source = df.loc[original_idx, 'source']
            
            for idx in group.index[1:]:
                if df.loc[idx, 'source'] != original_source:  # Different source = copy
                    df.loc[idx, 'is_original'] = False
                    df.loc[idx, 'duplicate_of'] = original_idx
                    df.loc[idx, 'similarity_score'] = 1.0  # Exact match
                    df.loc[idx, 'original_source'] = original_source
                    df.loc[idx, 'similarity_category'] = 'exact_copy'
                    exact_duplicates += 1
    
    print(f"    Found {exact_duplicates} exact copies (same title, different source)")
    
    # Step 2: Find semantic duplicates using embeddings
    print("  Step 2: Finding semantic duplicates (similar content, different title)...")
    
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Only process articles that are still marked as original
        originals_mask = df['is_original'] == True
        originals_df = df[originals_mask]
        
        if len(originals_df) > 1:
            # Create text for embedding (title + first 500 chars of text)
            texts = (originals_df['title'].fillna('') + ' ' + 
                    originals_df['text'].fillna('').str[:500]).tolist()
            
            # Load model and compute embeddings
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
            print(f"    Computing embeddings for {len(texts)} articles...")
            embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
            
            # Compute similarity matrix
            sim_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(sim_matrix, 0)  # Ignore self-similarity
            
            # Find semantic duplicates
            semantic_duplicates = 0
            similar_articles = 0
            original_indices = originals_df.index.tolist()
            
            for i, idx_i in enumerate(original_indices):
                for j, idx_j in enumerate(original_indices):
                    if i >= j:  # Skip self and already processed pairs
                        continue
                    
                    similarity = sim_matrix[i, j]
                    if similarity >= similarity_threshold:
                        # Different titles but similar content
                        title_i = df.loc[idx_i, 'title']
                        title_j = df.loc[idx_j, 'title']
                        
                        if title_i != title_j:  # Only if titles are different
                            # Determine which is original by date
                            date_i = df.loc[idx_i, 'parsed_date']
                            date_j = df.loc[idx_j, 'parsed_date']
                            
                            if pd.isna(date_i) and pd.isna(date_j):
                                continue
                            elif pd.isna(date_j) or (not pd.isna(date_i) and date_i <= date_j):
                                original_idx, copy_idx = idx_i, idx_j
                            else:
                                original_idx, copy_idx = idx_j, idx_i
                            
                            # Only mark as copy if from different source
                            if df.loc[original_idx, 'source'] != df.loc[copy_idx, 'source']:
                                if df.loc[copy_idx, 'is_original']:  # Not already marked
                                    df.loc[copy_idx, 'is_original'] = False
                                    df.loc[copy_idx, 'duplicate_of'] = original_idx
                                    df.loc[copy_idx, 'similarity_score'] = float(similarity)
                                    df.loc[copy_idx, 'original_source'] = df.loc[original_idx, 'source']
                                    
                                    if similarity >= 0.85:
                                        df.loc[copy_idx, 'similarity_category'] = 'semantic_copy'
                                        semantic_duplicates += 1
                                    else:
                                        df.loc[copy_idx, 'similarity_category'] = 'paraphrased'
                                        similar_articles += 1
            
            print(f"    Found {semantic_duplicates} semantic copies (>85% similar)")
            print(f"    Found {similar_articles} paraphrased articles (70-85% similar)")
        
    except ImportError:
        print("    ⚠️ sentence-transformers not installed, skipping semantic analysis")
        print("    Install with: pip install sentence-transformers")
    
    # Step 3: Calculate spread statistics
    print("  Step 3: Calculating spread statistics...")
    
    # Count how many copies each original has
    spread_counts = df[df['is_original'] == False].groupby('duplicate_of').size()
    df['spread_count'] = df.index.map(lambda x: spread_counts.get(x, 0))
    
    # Summary
    total = len(df)
    originals = df['is_original'].sum()
    copies = total - originals
    
    print(f"\n📊 Spread Analysis Summary:")
    print(f"   Total articles: {total}")
    print(f"   Original articles: {originals} ({originals/total*100:.1f}%)")
    print(f"   Copies/duplicates: {copies} ({copies/total*100:.1f}%)")
    
    # Most copied stories
    if spread_counts.any():
        top_spread = spread_counts.nlargest(3)
        print(f"\n   Most copied stories:")
        for idx, count in top_spread.items():
            title = df.loc[idx, 'title'][:50]
            source = df.loc[idx, 'source']
            print(f"     - '{title}...' ({source}) → {count} copies")
    
    # Drop helper column
    df = df.drop(columns=['parsed_date'])
    
    return df


def clean_articles(input_file='data/raw/articles_scraped.json', output_file='data/processed/articles.csv', incremental=True, detect_spread=True):
    """
    Load articles from JSON, clean text, and save to CSV.
    Supports incremental mode to preserve existing articles.
    
    Args:
        input_file: Path to input JSON file (defaults to articles_scraped.json with full content)
        output_file: Path to output CSV file
        incremental: If True, merge with existing articles.csv and only add new ones
        detect_spread: If True, detect duplicates and track news spread (keeps all articles)
    
    Returns:
        pandas.DataFrame: Cleaned articles with spread analysis
    """
    # Use scraped articles by default (has full text content)
    if input_file == 'data/raw/articles_scraped.json' and not Path(input_file).exists():
        # Fallback to most recent JSON if scraped file doesn't exist
        json_files = glob.glob('data/raw/news_*.json')
        if not json_files:
            raise FileNotFoundError("No JSON files found in data/raw/")
        input_file = max(json_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"Warning: articles_scraped.json not found, using fallback: {input_file}")
    
    print(f"Using input file: {input_file}")
    
    # Load existing articles if in incremental mode
    existing_df = pd.DataFrame()
    if incremental and Path(output_file).exists():
        print(f"Loading existing articles from {output_file}...")
        existing_df = pd.read_csv(output_file)
        print(f"Found {len(existing_df)} existing articles")
    
    # Load JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list format and dict with 'articles' key
    if isinstance(data, list):
        articles = data
    else:
        articles = data.get('articles', [])
    
    # Process each article
    processed_articles = []
    skipped_paid = 0
    skipped_short = 0
    
    for article in articles:
        # Get title and text
        title = article.get('title', '') or ''
        
        # Try 'text' field first (scraped content), then fallback to description+content (API)
        text = article.get('text', '')
        if not text or text == "ONLY AVAILABLE IN PAID PLANS":
            description = article.get('description', '') or ''
            content = article.get('content', '') or ''
            combined_text = f"{title} {description} {content}"
        else:
            combined_text = text  # Use full scraped text directly
        
        # Skip articles with "ONLY AVAILABLE IN PAID PLANS" or insufficient content
        if "ONLY AVAILABLE IN PAID PLANS" in combined_text:
            skipped_paid += 1
            continue
        
        # Clean text
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', combined_text)
        # Remove URLs
        clean_text = re.sub(r'http[s]?://\S+|www\.\S+', '', clean_text)
        # Remove multiple spaces
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Only add articles with substantial content (minimum 100 words for good summaries)
        word_count = len(clean_text.split())
        if clean_text and word_count >= 100:
            processed_articles.append({
                'title': title,
                'text': clean_text,
                'publishedAt': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', '') if isinstance(article.get('source'), dict) else article.get('source', '')
            })
        else:
            skipped_short += 1
    
    # Create DataFrame
    df = pd.DataFrame(processed_articles)
    
    # Merge with existing articles if incremental mode
    if incremental and not existing_df.empty:
        print(f"Merging {len(df)} new articles with {len(existing_df)} existing articles...")
        df = pd.concat([existing_df, df], ignore_index=True)
    
    # Remove exact duplicates (same title + publishedAt + source) - these are true duplicates
    # But KEEP articles with same title from different sources - these are news spread
    initial_count = len(df)
    df = df.drop_duplicates(subset=['title', 'publishedAt', 'source'], keep='last')
    exact_duplicates_removed = initial_count - len(df)
    
    # Detect news spread if enabled
    if detect_spread:
        df = detect_duplicates_and_spread(df, similarity_threshold=0.70)
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    if skipped_paid > 0:
        print(f"Skipped {skipped_paid} articles with restricted/paid content")
    if skipped_short > 0:
        print(f"Skipped {skipped_short} articles with insufficient content (< 100 words)")
    if exact_duplicates_removed > 0:
        print(f"Removed {exact_duplicates_removed} exact duplicate entries (same title+date+source)")
    if incremental and not existing_df.empty:
        new_articles = len(df) - len(existing_df)
        print(f"✅ Added {new_articles} new unique articles to existing {len(existing_df)}")
    print(f"Successfully processed {len(df)} articles total")
    return df


if __name__ == '__main__':
    df = clean_articles()
    print(f"Processed {len(df)} articles")
    print(df.head())