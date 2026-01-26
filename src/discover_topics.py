import os
import re
import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pyfunc
from datetime import datetime

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from umap import UMAP
from hdbscan import HDBSCAN


def _clean_text(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def _make_stopwords():
    extra = {
        "said",
        "says",
        "say",
        "mr",
        "mrs",
        "ms",
        "reuters",
        "ap",
        "associated",
        "press",
        "update",
        "updated",
        "breaking",
        "news",
        "report",
        "reports",
        "photo",
        "video",
        "caption",
        "copyright",
        "newsletter",
        # days/months often dominate news text
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        # extra generic words that can creep into labels
        "told",
        "tells",
        "would",
        "could",
        "also",
    }
    return set(ENGLISH_STOP_WORDS).union(extra)


def _build_model(stopwords):
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        device="cpu",
    )

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=list(stopwords),
        min_df=3,
        max_df=0.75,  # stricter -> removes very common terms in your corpus
        max_features=50000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # Reduce outliers by smoothing the space a bit
    umap_model = UMAP(
        n_neighbors=50,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # Softer clustering -> fewer -1, but still stable
    hdbscan_model = HDBSCAN(
        min_cluster_size=8,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    representation_model = [
        KeyBERTInspired(top_n_words=15),
        MaximalMarginalRelevance(diversity=0.3),
    ]

    return BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        language="english",
        verbose=True,
        min_topic_size=5,
        nr_topics="auto",
        top_n_words=10,
        calculate_probabilities=True,
    )


def discover_topics(
    input_file="data/processed/articles_with_sentiment.csv",
    output_dir="models/topic_model",
    text_col="text",
    title_col="title",
    use_title_and_lead=True,
    lead_chars=1200,
    mlflow_experiment_name="topic-modeling",
):
    # Start MLflow run
    mlflow.set_experiment(mlflow_experiment_name)
    
    num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    print(f"Using CPU with {num_threads} threads")

    df = pd.read_csv(input_file)

    if text_col not in df.columns:
        raise ValueError(
            f"Missing column '{text_col}'. Available columns: {list(df.columns)}"
        )

    df[text_col] = df[text_col].map(_clean_text)
    if title_col in df.columns:
        df[title_col] = df[title_col].map(_clean_text)

    before = len(df)
    df = df[df[text_col].str.len() > 0].copy()
    print(f"Dropped empty docs: {before - len(df)}")

    if use_title_and_lead and title_col in df.columns:
        texts = (
            df[title_col].fillna("").astype(str)
            + "\n\n"
            + df[text_col].astype(str).str.slice(0, lead_chars)
        ).map(_clean_text).tolist()
    else:
        texts = (
            df[text_col]
            .astype(str)
            .str.slice(0, lead_chars)
            .map(_clean_text)
            .tolist()
        )

    keep = [i for i, t in enumerate(texts) if len(t) > 0]
    df = df.iloc[keep].reset_index(drop=True)
    texts = [texts[i] for i in keep]
    print(f"Docs used: {len(texts)}")

    stopwords = _make_stopwords()
    topic_model = _build_model(stopwords)

    # Start MLflow run
    with mlflow.start_run(run_name=f"bertopic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("input_file", input_file)
        mlflow.log_param("text_col", text_col)
        mlflow.log_param("title_col", title_col)
        mlflow.log_param("use_title_and_lead", use_title_and_lead)
        mlflow.log_param("lead_chars", lead_chars)
        mlflow.log_param("num_documents", len(texts))
        mlflow.log_param("min_cluster_size", 8)
        mlflow.log_param("min_samples", 2)
        mlflow.log_param("min_topic_size", 5)
        mlflow.log_param("umap_n_neighbors", 40)
        mlflow.log_param("umap_n_components", 5)
        mlflow.log_param("embedding_model", "sentence-transformers/all-mpnet-base-v2")

        print("Training topic model...")
        topics, probs = topic_model.fit_transform(texts)

        outlier_ratio = float(np.mean(np.array(topics) == -1))
        print(f"Outlier ratio: {outlier_ratio:.2%}")
        print(topic_model.get_topic_info().head(12))

        # Log metrics
        num_topics = len([t for t in topic_model.get_topics().keys() if t != -1])
        mlflow.log_metric("num_topics", num_topics)
        mlflow.log_metric("outlier_ratio", outlier_ratio)
        mlflow.log_metric("num_outliers", int(np.sum(np.array(topics) == -1)))
        
        # Calculate and log topic distribution metrics
        topic_counts = pd.Series(topics).value_counts()
        mlflow.log_metric("largest_topic_size", int(topic_counts.iloc[0]))
        mlflow.log_metric("smallest_topic_size", int(topic_counts.iloc[-1]))
        mlflow.log_metric("avg_topic_size", float(topic_counts.mean()))

        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "bertopic_model")
        topic_model.save(model_path, serialization="pytorch")
        print(f"Saved model to {model_path}")
        
        # Log model to MLflow
        mlflow.log_artifact(model_path)
        
        # Log topic info as artifact
        topic_info_path = os.path.join(output_dir, "topic_info.csv")
        topic_model.get_topic_info().to_csv(topic_info_path, index=False)
        mlflow.log_artifact(topic_info_path)

    df["topic"] = topics

    label_stop = stopwords.union({"year", "years", "new", "one", "two", "three"})
    topic_labels = {-1: "Outlier"}
    for topic_id in topic_model.get_topics().keys():
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id) or []
        filtered = [w for w, _ in words if w.lower() not in label_stop]
        topic_labels[topic_id] = " / ".join(filtered[:3]) if filtered else f"Topic {topic_id}"

        df["topic_label"] = df["topic"].map(topic_labels)

        output_csv = "data/processed/articles_with_topics.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved topics to {output_csv}")
        
        # Log output CSV as artifact
        mlflow.log_artifact(output_csv)

        print("\nTop topics:")
        counts = df["topic"].value_counts().head(20)
        for tid, cnt in counts.items():
            print(f"{tid:>4}: {topic_labels.get(tid, str(tid))} ({cnt})")
        
        # Log top topics as text artifact
        top_topics_path = os.path.join(output_dir, "top_topics.txt")
        with open(top_topics_path, "w") as f:
            for tid, cnt in counts.items():
                f.write(f"{tid:>4}: {topic_labels.get(tid, str(tid))} ({cnt})\n")
        mlflow.log_artifact(top_topics_path)
        
        print(f"\nMLflow run completed. Experiment: {mlflow_experiment_name}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

    return topic_model, topics, df


if __name__ == "__main__":
    discover_topics(
        input_file="data/processed/articles_with_sentiment.csv",
        output_dir="models/topic_model",
        text_col="text",
        title_col="title",
        use_title_and_lead=True,
        lead_chars=1200,
    )