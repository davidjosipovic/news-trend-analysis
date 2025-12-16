import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Page config
st.set_page_config(page_title="News Trend Analysis", layout="wide")

# Title
st.title("📊 News Trend Analysis Dashboard")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📰 News Analysis", "🔮 Predictive Analytics", "🔍 Duplicate Analysis"])

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

# Tab 1: News Analysis (original dashboard)
with tab1:
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
            help="Same news story covered by multiple sources. When disabled, only original articles are shown (copies/duplicates hidden)."
        )
    
        # Apply filters
        filtered_df = df.copy()
        if 'sentiment' in df.columns and selected_sentiment != 'All':
            filtered_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]
        if 'topic_name' in df.columns and selected_topic != 'All':
            filtered_df = filtered_df[filtered_df['topic_name'] == selected_topic]
    
        # Filter duplicates based on spread analysis if available
        if not show_duplicates:
            if 'is_original' in filtered_df.columns:
                # Use spread analysis - show only original articles
                chart_df = filtered_df[filtered_df['is_original'] == True].copy()
            elif 'title' in filtered_df.columns:
                # Fallback: simple deduplication by title
                chart_df = filtered_df.drop_duplicates(subset=['title']).copy()
            else:
                chart_df = filtered_df.copy()
        else:
            chart_df = filtered_df.copy()
    
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
    
        # Calculate counts with spread analysis
        total_count = len(filtered_df)  # Total including copies
        
        if 'is_original' in filtered_df.columns:
            # Use spread analysis for accurate counts
            original_count = len(filtered_df[filtered_df['is_original'] == True])
            copies_count = total_count - original_count
        else:
            # Fallback: simple title deduplication
            original_count = len(filtered_df.drop_duplicates(subset=['title'])) if 'title' in filtered_df.columns else total_count
            copies_count = total_count - original_count
    
        col1.metric("Total Articles", total_count)
        col2.metric("Original Articles", original_count,
                   help="Articles that are first to publish this story")
        col3.metric("Copies/Duplicates", copies_count,
                   help="Articles copying the same story from other sources")
        
        if 'sentiment' in chart_df.columns:
            most_common_sent = chart_df['sentiment'].mode()[0] if len(chart_df) > 0 else "N/A"
            col4.metric("Most Common Sentiment", most_common_sent)
        
        # Show info about filtering
        if not show_duplicates and copies_count > 0:
            st.info(f"📊 Showing **{original_count} original articles** ({copies_count} copies hidden). Toggle 'Show duplicate articles' in sidebar to see all {total_count} articles including copies from different sources.")
    
        # Charts
        col1, col2 = st.columns(2)
    
        # Bar chart - articles by topic (using chart_df for unique/all data)
        with col1:
            st.subheader("📈 Articles by Topic")
            if 'topic_name' in chart_df.columns:
                topic_counts = chart_df['topic_name'].value_counts().reset_index()
                topic_counts.columns = ['topic', 'count']
                fig_bar = px.bar(topic_counts, x='topic', y='count', 
                               color='count', color_continuous_scale='Blues')
                st.plotly_chart(fig_bar, use_container_width=True)
    
        # Pie chart - sentiment distribution (using chart_df for unique/all data)
        with col2:
            st.subheader("🎯 Sentiment Distribution")
            if 'sentiment' in chart_df.columns:
                sentiment_counts = chart_df['sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['sentiment', 'count']
                fig_pie = px.pie(sentiment_counts, names='sentiment', values='count',
                               color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig_pie, use_container_width=True)
    
        # Line chart - sentiment over time (using chart_df for unique/all data)
        st.subheader("📅 Sentiment Over Time")
        date_col = 'publishedAt' if 'publishedAt' in chart_df.columns else 'date'
        if date_col in chart_df.columns and 'sentiment' in chart_df.columns:
            # Parse date if not already datetime
            if not pd.api.types.is_datetime64_any_dtype(chart_df[date_col]):
                chart_df[date_col] = pd.to_datetime(chart_df[date_col], errors='coerce')
        
            # Group by date (without time) and sentiment
            chart_df['date_only'] = chart_df[date_col].dt.date
            sentiment_time = chart_df.groupby(['date_only', 'sentiment']).size().reset_index(name='count')
        
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
        
            # Filter duplicates based on show_duplicates setting
            if not show_duplicates:
                if 'is_original' in display_df.columns:
                    # Use spread analysis - show only originals
                    display_df = display_df[display_df['is_original'] == True]
                else:
                    # Fallback: deduplicate by title
                    display_df = display_df.drop_duplicates(subset=['title'])
        
            display_df = display_df.reset_index(drop=True)
            total_articles = len(display_df)
        
            if total_articles == 0:
                st.info("No articles with summaries found for the selected filters.")
            else:
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
                        # Show spread info if available
                        if 'is_original' in row and 'similarity_category' in row:
                            is_orig = row['is_original']
                            category = row['similarity_category']
                            
                            if not is_orig:
                                orig_src = row.get('original_source', 'Unknown')
                                sim_score = row.get('similarity_score', 0)
                                
                                if category == 'exact_copy':
                                    st.warning(f"🔄 **Exact Copy** - Same title as article from {orig_src}")
                                elif category == 'semantic_copy':
                                    st.warning(f"📋 **Semantic Copy** - {sim_score:.0%} similar to article from {orig_src}")
                                elif category == 'paraphrased':
                                    st.info(f"✏️ **Paraphrased** - {sim_score:.0%} similar to article from {orig_src}")
                            elif 'spread_count' in row and row['spread_count'] > 0:
                                st.success(f"⭐ **Original Article** - Copied by {int(row['spread_count'])} other sources")
                        
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

# Tab 2: Predictive Analytics
with tab2:
    try:
        from dashboard.predictive_components import add_predictive_tab_to_dashboard
        add_predictive_tab_to_dashboard()
    except ImportError as e:
        st.warning(f"⚠️ Predictive analytics module not available: {e}")
        st.info("To enable predictive analytics:")
        st.code("pip install xgboost imbalanced-learn optuna fastapi", language="bash")
        st.code("python train_models.py", language="bash")
    except Exception as e:
        st.error(f"❌ Error loading predictive analytics: {e}")

# Tab 3: Duplicate Analysis
with tab3:
    try:
        from dashboard.duplicate_components import add_duplicate_analysis_to_dashboard
        add_duplicate_analysis_to_dashboard()
    except ImportError as e:
        st.warning(f"⚠️ Duplicate analysis module not available: {e}")
        st.info("To enable duplicate analysis:")
        st.code("pip install sentence-transformers", language="bash")
        st.code("python -c \"from analysis.duplicate_detector import DuplicateAnalysis; DuplicateAnalysis().analyze_file('data/processed/articles_with_topics.csv')\"", language="bash")
    except Exception as e:
        st.error(f"❌ Error loading duplicate analysis: {e}")
