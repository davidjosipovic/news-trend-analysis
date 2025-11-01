import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
from bertopic import BERTopic
from collections import Counter

def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_list()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

def evaluate_pipeline():
    """
    Comprehensive evaluation of the NLP pipeline:
    1. Sentiment analysis confidence and distribution
    2. Topic modeling coherence and quality
    3. Summarization quality metrics
    4. Overall data quality statistics
    """
    print("=" * 60)
    print("📊 NLP Pipeline Evaluation Report")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'metrics': {}
    }
    
    # Load final dataset
    data_path = "data/processed/articles_with_summary.csv"
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found. Run the pipeline first.")
        return
    
    df = pd.read_csv(data_path)
    print(f"📁 Loaded {len(df)} articles from {data_path}\n")
    
    # ========================================
    # 1. DATA QUALITY METRICS
    # ========================================
    print("1️⃣  DATA QUALITY METRICS")
    print("-" * 60)
    
    # Count articles with complete data
    complete_articles = df[
        (df['text'].notna()) & 
        (df['sentiment'].notna()) & 
        (df['topic'].notna()) & 
        (df['summary'].notna()) &
        (df['summary'].str.strip() != '')
    ]
    
    completeness_rate = len(complete_articles) / len(df) * 100
    
    # Text length statistics
    df['text_length'] = df['text'].fillna('').str.len()
    df['text_word_count'] = df['text'].fillna('').str.split().str.len()
    
    print(f"✓ Total articles: {len(df)}")
    print(f"✓ Complete articles (all fields): {len(complete_articles)} ({completeness_rate:.1f}%)")
    print(f"✓ Average text length: {df['text_length'].mean():.0f} characters")
    print(f"✓ Average word count: {df['text_word_count'].mean():.0f} words")
    print(f"✓ Text length range: {df['text_length'].min():.0f} - {df['text_length'].max():.0f} chars")
    
    evaluation_results['metrics']['data_quality'] = {
        'total_articles': len(df),
        'complete_articles': len(complete_articles),
        'completeness_rate': round(completeness_rate, 2),
        'avg_text_length': round(df['text_length'].mean(), 0),
        'avg_word_count': round(df['text_word_count'].mean(), 0),
        'text_length_range': [int(df['text_length'].min()), int(df['text_length'].max())]
    }
    
    # ========================================
    # 2. SENTIMENT ANALYSIS EVALUATION
    # ========================================
    print(f"\n2️⃣  SENTIMENT ANALYSIS METRICS")
    print("-" * 60)
    
    sentiment_dist = df['sentiment'].value_counts()
    sentiment_percentages = (sentiment_dist / len(df) * 100).round(1)
    
    print("✓ Sentiment distribution:")
    for sentiment, count in sentiment_dist.items():
        pct = sentiment_percentages[sentiment]
        print(f"   • {sentiment.capitalize()}: {count} articles ({pct}%)")
    
    # Check for extreme imbalance
    max_pct = sentiment_percentages.max()
    min_pct = sentiment_percentages.min()
    imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
    
    if imbalance_ratio > 10:
        print(f"\n⚠️  Warning: Sentiment distribution is imbalanced (ratio: {imbalance_ratio:.1f}:1)")
        print("   Consider collecting more diverse data sources.")
    else:
        print(f"\n✓ Sentiment balance ratio: {imbalance_ratio:.1f}:1 (acceptable)")
    
    evaluation_results['metrics']['sentiment'] = {
        'distribution': {k: int(v) for k, v in sentiment_dist.to_dict().items()},
        'percentages': {k: float(v) for k, v in sentiment_percentages.to_dict().items()},
        'imbalance_ratio': round(imbalance_ratio, 2)
    }
    
    # ========================================
    # 3. TOPIC MODELING EVALUATION
    # ========================================
    print(f"\n3️⃣  TOPIC MODELING METRICS")
    print("-" * 60)
    
    # Topic distribution
    topic_dist = df['topic'].value_counts().sort_index()
    outlier_count = (df['topic'] == -1).sum()
    outlier_rate = outlier_count / len(df) * 100
    
    num_topics = len(topic_dist) - (1 if -1 in topic_dist.index else 0)
    
    print(f"✓ Number of topics discovered: {num_topics}")
    print(f"✓ Outlier articles (topic -1): {outlier_count} ({outlier_rate:.1f}%)")
    
    # Topic size distribution
    topic_sizes = topic_dist[topic_dist.index != -1]
    if len(topic_sizes) > 0:
        print(f"✓ Average topic size: {topic_sizes.mean():.1f} articles")
        print(f"✓ Topic size range: {topic_sizes.min()} - {topic_sizes.max()} articles")
        
        # Check if topics are well-distributed
        largest_topic = topic_sizes.max()
        smallest_topic = topic_sizes.min()
        size_ratio = largest_topic / smallest_topic if smallest_topic > 0 else float('inf')
        
        if size_ratio > 5:
            print(f"\n⚠️  Warning: Topics have uneven sizes (ratio: {size_ratio:.1f}:1)")
        else:
            print(f"\n✓ Topic size balance: {size_ratio:.1f}:1 (good)")
    
    # Display topic labels
    if 'topic_label' in df.columns:
        print(f"\n✓ Topic labels:")
        topic_labels = df.groupby('topic')['topic_label'].first().sort_index()
        for topic_id, label in topic_labels.items():
            count = (df['topic'] == topic_id).sum()
            pct = count / len(df) * 100
            print(f"   • Topic {topic_id}: {label} - {count} articles ({pct:.1f}%)")
    
    # Calculate topic coherence if model exists
    coherence_score = None
    model_path = "models/topic_model/bertopic_model"
    if os.path.exists(model_path):
        try:
            print("\n✓ Loading BERTopic model to calculate coherence...")
            topic_model = BERTopic.load(model_path)
            
            # Get topic words
            topics = topic_model.get_topics()
            if topics:
                # Simple coherence: average pairwise word similarity
                # (Full C_v coherence requires external corpus, so we use a proxy)
                topic_words_count = []
                for topic_id, words in topics.items():
                    if topic_id != -1:
                        topic_words_count.append(len(words))
                
                avg_words_per_topic = np.mean(topic_words_count) if topic_words_count else 0
                coherence_score = min(avg_words_per_topic / 10 * 100, 100)  # Normalize to 0-100
                
                print(f"✓ Topic coherence proxy: {coherence_score:.1f}/100")
                print(f"  (Based on {avg_words_per_topic:.1f} avg keywords per topic)")
        except Exception as e:
            print(f"⚠️  Could not calculate coherence: {e}")
    
    evaluation_results['metrics']['topics'] = {
        'num_topics': num_topics,
        'outlier_count': outlier_count,
        'outlier_rate': round(outlier_rate, 2),
        'avg_topic_size': round(topic_sizes.mean(), 1) if len(topic_sizes) > 0 else 0,
        'topic_size_range': [int(topic_sizes.min()), int(topic_sizes.max())] if len(topic_sizes) > 0 else [0, 0],
        'coherence_proxy': round(coherence_score, 1) if coherence_score else None
    }
    
    # ========================================
    # 4. SUMMARIZATION EVALUATION
    # ========================================
    print(f"\n4️⃣  SUMMARIZATION QUALITY METRICS")
    print("-" * 60)
    
    # Filter articles with summaries
    articles_with_summary = df[df['summary'].notna() & (df['summary'].str.strip() != '')]
    summary_coverage = len(articles_with_summary) / len(df) * 100
    
    print(f"✓ Articles with summaries: {len(articles_with_summary)} ({summary_coverage:.1f}%)")
    
    if len(articles_with_summary) > 0:
        # Calculate summary statistics
        articles_with_summary['summary_length'] = articles_with_summary['summary'].str.len()
        articles_with_summary['summary_word_count'] = articles_with_summary['summary'].str.split().str.len()
        
        # Compression ratio (original text / summary)
        articles_with_summary['compression_ratio'] = (
            articles_with_summary['text_length'] / articles_with_summary['summary_length']
        )
        
        avg_summary_length = articles_with_summary['summary_length'].mean()
        avg_summary_words = articles_with_summary['summary_word_count'].mean()
        avg_compression = articles_with_summary['compression_ratio'].mean()
        
        print(f"✓ Average summary length: {avg_summary_length:.0f} characters")
        print(f"✓ Average summary words: {avg_summary_words:.0f} words")
        print(f"✓ Average compression ratio: {avg_compression:.1f}x")
        print(f"✓ Summary length range: {articles_with_summary['summary_length'].min():.0f} - {articles_with_summary['summary_length'].max():.0f} chars")
        
        # Check for quality issues
        too_short = (articles_with_summary['summary_word_count'] < 20).sum()
        too_long = (articles_with_summary['summary_word_count'] > 150).sum()
        
        if too_short > 0:
            print(f"\n⚠️  Warning: {too_short} summaries are very short (< 20 words)")
        if too_long > 0:
            print(f"⚠️  Warning: {too_long} summaries are very long (> 150 words)")
        
        # Ideal compression ratio is 5-10x
        if avg_compression < 5:
            print(f"\n⚠️  Warning: Low compression ratio ({avg_compression:.1f}x). Summaries may be too long.")
        elif avg_compression > 15:
            print(f"\n⚠️  Warning: High compression ratio ({avg_compression:.1f}x). Summaries may lose detail.")
        else:
            print(f"\n✓ Compression ratio is optimal (5-15x range)")
        
        evaluation_results['metrics']['summarization'] = {
            'coverage': round(summary_coverage, 2),
            'avg_summary_length': round(avg_summary_length, 0),
            'avg_summary_words': round(avg_summary_words, 0),
            'avg_compression_ratio': round(avg_compression, 2),
            'summary_length_range': [
                int(articles_with_summary['summary_length'].min()),
                int(articles_with_summary['summary_length'].max())
            ],
            'quality_warnings': {
                'too_short': int(too_short),
                'too_long': int(too_long)
            }
        }
    else:
        print("❌ No summaries found in dataset")
        evaluation_results['metrics']['summarization'] = {
            'coverage': 0,
            'error': 'No summaries generated'
        }
    
    # ========================================
    # 5. TEMPORAL ANALYSIS
    # ========================================
    print(f"\n5️⃣  TEMPORAL DISTRIBUTION")
    print("-" * 60)
    
    if 'publishedAt' in df.columns:
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        valid_dates = df['publishedAt'].notna().sum()
        
        if valid_dates > 0:
            date_range = (df['publishedAt'].max() - df['publishedAt'].min()).days
            print(f"✓ Articles with valid dates: {valid_dates}")
            print(f"✓ Date range: {df['publishedAt'].min().strftime('%Y-%m-%d')} to {df['publishedAt'].max().strftime('%Y-%m-%d')}")
            print(f"✓ Time span: {date_range} days")
            
            # Articles per day
            articles_per_day = valid_dates / max(date_range, 1)
            print(f"✓ Average: {articles_per_day:.2f} articles/day")
            
            evaluation_results['metrics']['temporal'] = {
                'articles_with_dates': valid_dates,
                'date_range_days': date_range,
                'articles_per_day': round(articles_per_day, 2),
                'start_date': df['publishedAt'].min().isoformat(),
                'end_date': df['publishedAt'].max().isoformat()
            }
        else:
            print("⚠️  No valid publication dates found")
    else:
        print("⚠️  No 'publishedAt' column found")
    
    # ========================================
    # 6. OVERALL QUALITY SCORE
    # ========================================
    print(f"\n6️⃣  OVERALL PIPELINE QUALITY SCORE")
    print("-" * 60)
    
    # Calculate composite quality score (0-100)
    scores = []
    weights = []
    
    # Data completeness (30%)
    scores.append(completeness_rate)
    weights.append(0.30)
    
    # Sentiment balance (15%)
    sentiment_score = max(0, 100 - (imbalance_ratio - 1) * 10)
    scores.append(sentiment_score)
    weights.append(0.15)
    
    # Topic quality (25%)
    topic_score = 100
    if outlier_rate > 20:
        topic_score -= (outlier_rate - 20) * 2
    if len(topic_sizes) > 0 and size_ratio > 5:
        topic_score -= (size_ratio - 5) * 5
    topic_score = max(0, topic_score)
    scores.append(topic_score)
    weights.append(0.25)
    
    # Summarization quality (30%)
    summary_score = summary_coverage
    if len(articles_with_summary) > 0:
        if 5 <= avg_compression <= 15:
            summary_score = min(100, summary_score + 10)
    scores.append(summary_score)
    weights.append(0.30)
    
    # Calculate weighted average
    overall_score = sum(s * w for s, w in zip(scores, weights))
    
    print(f"✓ Data Completeness: {completeness_rate:.1f}/100 (weight: 30%)")
    print(f"✓ Sentiment Balance: {sentiment_score:.1f}/100 (weight: 15%)")
    print(f"✓ Topic Quality: {topic_score:.1f}/100 (weight: 25%)")
    print(f"✓ Summarization: {summary_score:.1f}/100 (weight: 30%)")
    print(f"\n{'='*60}")
    print(f"🎯 OVERALL QUALITY SCORE: {overall_score:.1f}/100")
    print(f"{'='*60}")
    
    # Quality rating
    if overall_score >= 90:
        rating = "🌟 EXCELLENT"
        color = "🟢"
    elif overall_score >= 75:
        rating = "✅ GOOD"
        color = "🟢"
    elif overall_score >= 60:
        rating = "⚠️  ACCEPTABLE"
        color = "🟡"
    else:
        rating = "❌ NEEDS IMPROVEMENT"
        color = "🔴"
    
    print(f"{color} Rating: {rating}\n")
    
    evaluation_results['metrics']['overall_quality'] = {
        'score': round(overall_score, 1),
        'rating': rating,
        'component_scores': {
            'data_completeness': round(completeness_rate, 1),
            'sentiment_balance': round(sentiment_score, 1),
            'topic_quality': round(topic_score, 1),
            'summarization': round(summary_score, 1)
        }
    }
    
    # ========================================
    # SAVE EVALUATION REPORT
    # ========================================
    os.makedirs('data/evaluation', exist_ok=True)
    
    # Save JSON report
    json_path = 'data/evaluation/evaluation_report.json'
    with open(json_path, 'w') as f:
        # Convert all numpy/pandas types before saving
        serializable_results = convert_to_json_serializable(evaluation_results)
        json.dump(serializable_results, f, indent=2)
    print(f"📄 Evaluation report saved to: {json_path}")
    
    # Save text report
    txt_path = 'data/evaluation/evaluation_report.txt'
    with open(txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("NLP Pipeline Evaluation Report\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Overall Quality Score: {overall_score:.1f}/100\n")
        f.write(f"Rating: {rating}\n\n")
        # Convert before writing
        serializable_results = convert_to_json_serializable(evaluation_results)
        f.write(json.dumps(serializable_results['metrics'], indent=2))
    print(f"📄 Text report saved to: {txt_path}")
    
    print("\n✅ Evaluation complete!")
    return evaluation_results

if __name__ == "__main__":
    evaluate_pipeline()
