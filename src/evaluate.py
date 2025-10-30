import mlflow
import os
from datetime import datetime
from collections import Counter

# Set MLflow tracking directory
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlflow_tracking')}")
mlflow.set_experiment("news-trend-analysis")


def log_bertopic_metrics(topic_model, documents):
    """
    Log BERTopic model metrics to MLflow.
    
    Args:
        topic_model: Trained BERTopic model
        documents: List of documents used for topic modeling
    """
    with mlflow.start_run():
        # Log number of topics (excluding outlier topic -1)
        num_topics = len(set(topic_model.topics_)) - (1 if -1 in topic_model.topics_ else 0)
        mlflow.log_metric("num_topics", num_topics)
        
        # Log total number of topics including outliers
        mlflow.log_metric("total_topics_with_outliers", len(set(topic_model.topics_)))
        
        # Log number of documents
        mlflow.log_metric("num_documents", len(documents))
        
        # Log processing time
        mlflow.log_param("processing_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        print(f"Logged BERTopic metrics: {num_topics} topics from {len(documents)} documents")


def log_sentiment_metrics(sentiments):
    """
    Log sentiment distribution metrics to MLflow.
    
    Args:
        sentiments: List of sentiment labels (e.g., ['positive', 'negative', 'neutral'])
    """
    with mlflow.start_run():
        # Calculate sentiment distribution
        sentiment_counts = Counter(sentiments)
        total = len(sentiments)
        
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total) * 100
            mlflow.log_metric(f"sentiment_{sentiment}_percentage", percentage)
            mlflow.log_metric(f"sentiment_{sentiment}_count", count)
        
        mlflow.log_metric("total_analyzed", total)
        mlflow.log_param("last_processed", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        print(f"Logged sentiment metrics: {dict(sentiment_counts)}")


def log_combined_metrics(topic_model, documents, sentiments):
    """
    Log combined BERTopic and sentiment metrics to MLflow.
    
    Args:
        topic_model: Trained BERTopic model
        documents: List of documents
        sentiments: List of sentiment labels
    """
    with mlflow.start_run(run_name="combined_analysis"):
        # BERTopic metrics
        num_topics = len(set(topic_model.topics_)) - (1 if -1 in topic_model.topics_ else 0)
        mlflow.log_metric("num_topics", num_topics)
        mlflow.log_metric("num_documents", len(documents))
        
        # Sentiment distribution
        sentiment_counts = Counter(sentiments)
        total = len(sentiments)
        
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total) * 100
            mlflow.log_metric(f"sentiment_{sentiment}_percentage", percentage)
        
        # Processing time
        mlflow.log_param("processing_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        print(f"Logged combined metrics: {num_topics} topics, {total} sentiments analyzed")


if __name__ == "__main__":
    # Example usage
    print(f"MLflow tracking directory: {os.path.abspath('mlflow_tracking')}")
    print("Use the functions above to log your metrics to MLflow")