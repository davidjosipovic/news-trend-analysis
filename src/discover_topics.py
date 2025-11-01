import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
import os
import torch

def discover_topics(input_file='data/processed/articles_with_sentiment.csv', output_dir='models/topic_model'):
    """
    Discover topics in articles using BERTopic clustering.
    Uses pre-trained sentence embeddings + HDBSCAN clustering (unsupervised).
    CPU-optimized with multi-threading.
    
    Args:
        input_file: Path to preprocessed articles CSV (should have sentiment column)
        output_dir: Directory to save the trained model
    """
    # Set CPU threads for optimal performance
    num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    print(f"Using CPU with {num_threads} threads")
    
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    texts = df['text'].tolist()
    
    print(f"Loaded {len(texts)} articles")
    
    # Initialize embedding model with CPU optimization
    print("Loading embedding model (optimized for CPU)...")
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device='cpu'
    )
    # Enable multi-process encoding for faster CPU processing
    embedding_model.encode(["test"], show_progress_bar=False)  # Warm up
    
    # Create automatic topic labeling representation
    print("Setting up automatic topic labeling...")
    representation_model = KeyBERTInspired()
    
    # Initialize BERTopic with automatic labeling
    print("Initializing BERTopic...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        representation_model=representation_model,
        language='english',
        verbose=True,
        min_topic_size=2  # Minimum 2 documents per topic
    )
    
    # Fit the model
    print("Training topic model...")
    topics, probs = topic_model.fit_transform(texts)
    
    # Display top 10 topics
    print("\nTop 10 Topics:")
    topic_info = topic_model.get_topic_info()
    print(topic_info.head(10))
    
    # Display topic words
    num_topics_to_show = min(10, len(topic_info) - 1)
    for topic_id in range(num_topics_to_show):
        if topic_id != -1:  # Skip outlier topic
            print(f"\nTopic {topic_id}:")
            print(topic_model.get_topic(topic_id))
    
    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'bertopic_model')
    topic_model.save(model_path, serialization="pytorch")
    
    print(f"\n✅ Model saved to {model_path}")
    
    # Add topic assignments to dataframe (PRESERVE all existing columns including sentiment)
    df['topic'] = topics
    
    # Add automatic topic labels with improved naming
    print("\n✅ Generating automatic topic labels...")
    topic_labels = {}
    topic_descriptions = {}
    
    # Define stop words to exclude from labels
    stop_words = {'we', 'us', 'our', 'they', 'their', 'this', 'that', 'these', 'those', 
                  'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
                  'said', 'says', 'say', 'year', 'years', 'new', 'also', 'been', 'being'}
    
    for topic_id in topic_model.get_topics().keys():
        if topic_id != -1:  # Skip outlier
            # Get top words for topic
            words = topic_model.get_topic(topic_id)
            
            # Filter out stop words and select meaningful keywords
            filtered_words = [w[0] for w in words if w[0].lower() not in stop_words][:5]
            
            # Create concise label from top 2-3 keywords
            if len(filtered_words) >= 3:
                # Use top 3 meaningful words
                label = "_".join(filtered_words[:3]).title()
            elif len(filtered_words) >= 2:
                # Use top 2 if we have them
                label = "_".join(filtered_words[:2]).title()
            else:
                # Fallback to original if filtering removed too much
                label = "_".join([w[0] for w in words[:3]]).title()
            
            # Create human-readable description
            top_5_words = ", ".join(filtered_words[:5])
            topic_descriptions[topic_id] = f"{label} ({top_5_words})"
            topic_labels[topic_id] = label
        else:
            topic_labels[topic_id] = "Outlier"
            topic_descriptions[topic_id] = "Outlier (Mixed Topics)"
    
    df['topic_label'] = df['topic'].map(topic_labels)
    
    output_csv = 'data/processed/articles_with_topics.csv'
    df.to_csv(output_csv, index=False)
    print(f"✅ Topic assignments saved to {output_csv}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n✅ Improved topic labels:")
    for topic_id in sorted(topic_labels.keys()):
        count = (df['topic'] == topic_id).sum()
        description = topic_descriptions.get(topic_id, topic_labels[topic_id])
        print(f"   Topic {topic_id}: {description} - {count} articles")
    
    return topic_model, topics

if __name__ == "__main__":
    discover_topics()