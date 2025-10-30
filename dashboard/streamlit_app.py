import streamlit as st
import pandas as pd
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Analiza tržišnih trendova", layout="wide")

# Title
st.title("📊 Analiza tržišnih trendova kroz novinske članke")

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
    st.sidebar.header("Filteri")
    
    # Sentiment filter
    if 'sentiment' in df.columns:
        sentiments = ['Svi'] + list(df['sentiment'].unique())
        selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)
    
    # Topic filter
    if 'topic_name' in df.columns:
        topics = ['Svi'] + list(df['topic_name'].unique())
        selected_topic = st.sidebar.selectbox("Tema", topics)
    
    # Apply filters
    filtered_df = df.copy()
    if 'sentiment' in df.columns and selected_sentiment != 'Svi':
        filtered_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]
    if 'topic_name' in df.columns and selected_topic != 'Svi':
        filtered_df = filtered_df[filtered_df['topic_name'] == selected_topic]
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Ukupno članaka", len(filtered_df))
    if 'sentiment' in filtered_df.columns:
        col2.metric("Najčešći sentiment", filtered_df['sentiment'].mode()[0] if len(filtered_df) > 0 else "N/A")
    if 'topic_name' in filtered_df.columns:
        col3.metric("Najčešća tema", filtered_df['topic_name'].mode()[0] if len(filtered_df) > 0 else "N/A")
    
    # Charts
    col1, col2 = st.columns(2)
    
    # Bar chart - articles by topic
    with col1:
        st.subheader("📈 Broj članaka po temi")
        if 'topic_name' in filtered_df.columns:
            topic_counts = filtered_df['topic_name'].value_counts().reset_index()
            topic_counts.columns = ['tema', 'broj']
            fig_bar = px.bar(topic_counts, x='tema', y='broj', 
                           color='broj', color_continuous_scale='Blues')
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Pie chart - sentiment distribution
    with col2:
        st.subheader("🎯 Distribucija sentimenta")
        if 'sentiment' in filtered_df.columns:
            sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment', 'broj']
            fig_pie = px.pie(sentiment_counts, names='sentiment', values='broj',
                           color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Line chart - sentiment over time
    st.subheader("📅 Sentiment kroz vrijeme")
    if 'date' in filtered_df.columns and 'sentiment' in filtered_df.columns:
        sentiment_time = filtered_df.groupby(['date', 'sentiment']).size().reset_index(name='broj')
        fig_line = px.line(sentiment_time, x='date', y='broj', color='sentiment',
                         markers=True)
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Article summaries
    st.subheader("📰 Pregled članaka")
    if 'summary' in filtered_df.columns:
        # Remove duplicates based on title and drop rows with missing summaries
        display_df = filtered_df.drop_duplicates(subset=['title']).copy()
        
        for idx, row in display_df.iterrows():
            # Clean title - remove " - N/A" suffix
            title = str(row.get('title', 'Članak')).replace(' - N/A', '').strip()
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
                    st.write(f"**Tema:** {row['topic_name']}")
                if 'scraped' in row:
                    scrape_status = "✅ Scraped full text" if row['scraped'] else "⚠️ API content only"
                    st.write(f"**Source:** {scrape_status}")
                st.write(f"**Summary:** {summary_text}")

except FileNotFoundError:
    st.error("❌ Datoteka data/processed/articles_with_summary.csv nije pronađena.")
except Exception as e:
    st.error(f"❌ Greška pri učitavanju podataka: {str(e)}")