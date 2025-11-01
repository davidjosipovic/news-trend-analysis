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
    
    # Duplicate articles option
    st.sidebar.header("Display Options")
    show_duplicates = st.sidebar.checkbox(
        "Show duplicate articles from different sources", 
        value=False,
        help="Same news story covered by multiple sources"
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
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate unique articles count
    unique_count = len(filtered_df.drop_duplicates(subset=['title'])) if 'title' in filtered_df.columns else len(filtered_df)
    
    col1.metric("Total Articles", len(filtered_df))
    col2.metric("Unique Articles", unique_count)
    if 'sentiment' in filtered_df.columns:
        col3.metric("Most Common Sentiment", filtered_df['sentiment'].mode()[0] if len(filtered_df) > 0 else "N/A")
    if 'sentiment_confidence' in filtered_df.columns and len(filtered_df) > 0:
        avg_confidence = filtered_df['sentiment_confidence'].mean()
        col4.metric("Avg Confidence", f"{avg_confidence:.1%}")
    elif 'topic_name' in filtered_df.columns:
        col4.metric("Most Common Topic", filtered_df['topic_name'].mode()[0] if len(filtered_df) > 0 else "N/A")
    
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
    date_col = 'publishedAt' if 'publishedAt' in filtered_df.columns else 'date'
    if date_col in filtered_df.columns and 'sentiment' in filtered_df.columns:
        # Parse date if not already datetime
        if not pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
            filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
        
        # Group by date (without time) and sentiment
        filtered_df['date_only'] = filtered_df[date_col].dt.date
        sentiment_time = filtered_df.groupby(['date_only', 'sentiment']).size().reset_index(name='count')
        
        fig_line = px.line(sentiment_time, x='date_only', y='count', color='sentiment',
                         markers=True, 
                         color_discrete_map={'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'})
        fig_line.update_xaxes(title_text='Date')
        fig_line.update_yaxes(title_text='Number of Articles')
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("📊 Not enough date information to display sentiment over time.")
    
    # Article summaries with pagination
    st.subheader("📰 Article Overview")
    if 'summary' in filtered_df.columns:
        # Filter out articles without summaries first
        display_df = filtered_df[
            filtered_df['summary'].notna() & 
            (filtered_df['summary'].astype(str).str.strip() != '') &
            (filtered_df['summary'].astype(str).str.lower() != 'nan')
        ].copy()
        
        # Remove duplicates based on title (if option is disabled)
        if not show_duplicates:
            display_df = display_df.drop_duplicates(subset=['title'])
        
        display_df = display_df.reset_index(drop=True)
        total_articles = len(display_df)
        
        if total_articles == 0:
            st.info("No articles with summaries found for the selected filters.")
        else:
            # Show info about duplicates if they exist
            if not show_duplicates:
                total_with_dupes = len(filtered_df[
                    filtered_df['summary'].notna() & 
                    (filtered_df['summary'].astype(str).str.strip() != '') &
                    (filtered_df['summary'].astype(str).str.lower() != 'nan')
                ])
                if total_with_dupes > total_articles:
                    st.info(f"ℹ️ Showing {total_articles} unique articles ({total_with_dupes - total_articles} duplicates from other sources hidden). Enable 'Show duplicates' in sidebar to see all.")
            
            # Pagination settings
            articles_per_page = st.sidebar.number_input(
                "Articles per page", 
                min_value=5, 
                max_value=50, 
                value=10, 
                step=5
            )
            
            total_pages = (total_articles - 1) // articles_per_page + 1
            
            # Page selector
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page = st.selectbox(
                    f"Page (showing {total_articles} articles)", 
                    range(1, total_pages + 1),
                    format_func=lambda x: f"Page {x} of {total_pages}"
                )
            
            # Calculate start and end indices
            start_idx = (page - 1) * articles_per_page
            end_idx = min(start_idx + articles_per_page, total_articles)
            
            # Display articles for current page
            st.write(f"Showing articles {start_idx + 1} - {end_idx} of {total_articles}")
            
            for idx in range(start_idx, end_idx):
                row = display_df.iloc[idx]
                
                # Clean title - remove " - N/A" suffix
                title = str(row.get('title', 'Article')).replace(' - N/A', '').strip()
                date_str = row.get('publishedAt', row.get('date', 'N/A'))
                summary_text = str(row.get('summary', ''))
                    
                with st.expander(f"📄 {title} ({date_str})"):
                    if 'sentiment' in row:
                        sentiment_emoji = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}.get(row['sentiment'], '')
                        sentiment_text = f"**Sentiment:** {sentiment_emoji} {row['sentiment'].capitalize()}"
                        
                        # Add confidence badge if available
                        if 'sentiment_confidence' in row and pd.notna(row['sentiment_confidence']):
                            confidence = float(row['sentiment_confidence'])
                            confidence_pct = f"{confidence:.1%}"
                            
                            # Color code confidence
                            if confidence >= 0.8:
                                confidence_badge = f"🟢 {confidence_pct}"
                            elif confidence >= 0.6:
                                confidence_badge = f"🟡 {confidence_pct}"
                            else:
                                confidence_badge = f"🔴 {confidence_pct}"
                            
                            sentiment_text += f" (Confidence: {confidence_badge})"
                        
                        st.write(sentiment_text)
                    if 'topic_name' in row:
                        st.write(f"**Topic:** {row['topic_name']}")
                    if 'source' in row:
                        st.write(f"**Source:** {row['source']}")
                    st.write(f"**Summary:** {summary_text}")

except FileNotFoundError:
    st.error("❌ File data/processed/articles_with_summary.csv not found.")
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")