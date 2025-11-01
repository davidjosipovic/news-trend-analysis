import streamlit as st
import pandas as pd
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="News Trend Analysis", layout="wide")

# Title
st.title("📊 News Trend Analysis Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/articles_with_summary.csv")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Use automatic topic labels if available, otherwise map manually
    if 'topic_label' in df.columns:
        df['topic_name'] = df['topic_label']
    elif 'topic' in df.columns:
        # Fallback: manual mapping (if topic_label column doesn't exist)
        topic_names = {
            -1: 'Outlier (Mixed)',
            0: 'US-China Relations',
            1: 'Business & Tourism',
            2: 'Central Banks & Interest Rates',
            3: 'Agriculture & Fintech'
        }
        df['topic_name'] = df['topic'].map(topic_names).fillna(f'Topic {df["topic"]}')
    
    return df

try:
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Sentiment filter
    if 'sentiment' in df.columns:
        sentiments = ['All'] + list(df['sentiment'].unique())
        selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)
    
    # Topic filter
    if 'topic_name' in df.columns:
        topics = ['All'] + list(df['topic_name'].unique())
        selected_topic = st.sidebar.selectbox("Topic", topics)
    
    # Sort order
    st.sidebar.header("Sort Order")
    sort_order = st.sidebar.radio(
        "Sort articles by date:",
        ["Newest first", "Oldest first"],
        index=0
    )
    
    # Apply filters
    filtered_df = df.copy()
    if 'sentiment' in df.columns and selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]
    if 'topic_name' in df.columns and selected_topic != 'All':
        filtered_df = filtered_df[filtered_df['topic_name'] == selected_topic]
    
    # Apply sorting
    if 'publishedAt' in filtered_df.columns:
        filtered_df['publishedAt'] = pd.to_datetime(filtered_df['publishedAt'], errors='coerce')
        if sort_order == "Newest first":
            filtered_df = filtered_df.sort_values('publishedAt', ascending=False)
        else:
            filtered_df = filtered_df.sort_values('publishedAt', ascending=True)
    elif 'date' in filtered_df.columns:
        if sort_order == "Newest first":
            filtered_df = filtered_df.sort_values('date', ascending=False)
        else:
            filtered_df = filtered_df.sort_values('date', ascending=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Articles", len(filtered_df))
    if 'sentiment' in filtered_df.columns:
        col2.metric("Most Common Sentiment", filtered_df['sentiment'].mode()[0] if len(filtered_df) > 0 else "N/A")
    if 'topic_name' in filtered_df.columns:
        col3.metric("Most Common Topic", filtered_df['topic_name'].mode()[0] if len(filtered_df) > 0 else "N/A")
    
    # Charts
    col1, col2 = st.columns(2)
    
    # Bar chart - articles by topic
    with col1:
        st.subheader("📈 Articles by Topic")
        if 'topic_name' in filtered_df.columns:
            topic_counts = filtered_df['topic_name'].value_counts().reset_index()
            topic_counts.columns = ['topic', 'count']
            fig_bar = px.bar(topic_counts, x='topic', y='count', 
                           color='count', color_continuous_scale='Blues')
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Pie chart - sentiment distribution
    with col2:
        st.subheader("🎯 Sentiment Distribution")
        if 'sentiment' in filtered_df.columns:
            sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment', 'count']
            fig_pie = px.pie(sentiment_counts, names='sentiment', values='count',
                           color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Line chart - sentiment over time
    st.subheader("📅 Sentiment Over Time")
    if 'date' in filtered_df.columns and 'sentiment' in filtered_df.columns:
        sentiment_time = filtered_df.groupby(['date', 'sentiment']).size().reset_index(name='count')
        fig_line = px.line(sentiment_time, x='date', y='count', color='sentiment',
                         markers=True)
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Article summaries
    st.subheader("📰 Article Overview")
    if 'summary' in filtered_df.columns:
        # Remove duplicates based on title and drop rows with missing summaries
        display_df = filtered_df.drop_duplicates(subset=['title']).copy()
        
        for idx, row in display_df.iterrows():
            # Clean title - remove " - N/A" suffix
            title = str(row.get('title', 'Article')).replace(' - N/A', '').strip()
            date_str = row.get('publishedAt', row.get('date', 'N/A'))
            
            # Skip if summary is NaN/empty
            summary_text = str(row.get('summary', ''))
            if summary_text.lower() == 'nan' or not summary_text.strip():
                continue
                
            with st.expander(f"{title} ({date_str})"):
                if 'sentiment' in row:
                    sentiment_emoji = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}.get(row['sentiment'], '')
                    st.write(f"**Sentiment:** {sentiment_emoji} {row['sentiment'].capitalize()}")
                if 'topic_name' in row:
                    st.write(f"**Topic:** {row['topic_name']}")
                if 'scraped' in row:
                    scrape_status = "✅ Scraped full text" if row['scraped'] else "⚠️ API content only"
                    st.write(f"**Source:** {scrape_status}")
                st.write(f"**Summary:** {summary_text}")

except FileNotFoundError:
    st.error("❌ File data/processed/articles_with_summary.csv not found.")
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")