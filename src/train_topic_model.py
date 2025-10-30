import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
import os
import torch

def train_topic_model(input_file='data/processed/articles_with_sentiment.csv', output_dir='models/topic_model'):
    """
    Train BERTopic model on preprocessed articles.
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
    
    # Add automatic topic labels from model
    print("\n✅ Generating automatic topic labels...")
    topic_labels = {}
    for topic_id in topic_model.get_topics().keys():
        if topic_id != -1:  # Skip outlier
            # Get top words for topic
            words = topic_model.get_topic(topic_id)
            # Create label from top 3 keywords
            label = "_".join([w[0] for w in words[:3]]).title()
            topic_labels[topic_id] = label
        else:
            topic_labels[topic_id] = "Outlier"
    
    df['topic_label'] = df['topic'].map(topic_labels)
    
    output_csv = 'data/processed/articles_with_topics.csv'
    df.to_csv(output_csv, index=False)
    print(f"✅ Topic assignments saved to {output_csv}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n✅ Automatic topic labels:")
    for topic_id, label in sorted(topic_labels.items()):
        count = (df['topic'] == topic_id).sum()
        print(f"   Topic {topic_id}: {label} ({count} articles)")
    
    return topic_model, topics

if __name__ == "__main__":
    train_topic_model()