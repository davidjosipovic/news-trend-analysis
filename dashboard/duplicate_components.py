"""
Duplicate Detection Dashboard Components
========================================

Streamlit components for displaying duplicate detection results:
- Copy-paste journalism analysis
- Source originality ranking
- Duplicate visualization

Author: News Trend Analysis Team
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_duplicate_analysis():
    """Load articles with spread analysis from preprocessing."""
    try:
        # Load from processed articles (spread detection is now in preprocessing)
        for path in ['data/processed/articles.csv', 
                     'data/processed/articles_with_sentiment.csv',
                     'data/processed/articles_with_topics.csv']:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if 'is_original' in df.columns:
                    return df
        
        # If no spread data exists, need to rerun preprocessing
        st.warning("Spread analysis not found. Rerun: python src/preprocess_articles.py")
        return None
        
    except Exception as e:
        st.warning(f"⚠️ Could not load analysis: {e}")
        return None


def render_copy_paste_gauge(copy_paste_ratio: float):
    """Render gauge showing copy-paste journalism ratio."""
    
    # Determine color and label
    if copy_paste_ratio < 0.10:
        color = '#2ecc71'  # Green
        label = 'Excellent'
    elif copy_paste_ratio < 0.25:
        color = '#f1c40f'  # Yellow
        label = 'Moderate'
    elif copy_paste_ratio < 0.40:
        color = '#e67e22'  # Orange
        label = 'High'
    else:
        color = '#e74c3c'  # Red
        label = 'Very High'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=copy_paste_ratio * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Copy-Paste Ratio<br><span style='font-size:0.8em;color:gray'>{label}</span>"},
        number={'suffix': '%', 'font': {'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 10], 'color': '#d5f5e3'},
                {'range': [10, 25], 'color': '#fcf3cf'},
                {'range': [25, 40], 'color': '#fdebd0'},
                {'range': [40, 100], 'color': '#fadbd8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def render_source_originality_chart(df: pd.DataFrame):
    """Render bar chart comparing originality by source."""
    if df is None or 'source' not in df.columns or 'is_original' not in df.columns:
        return None
    
    # Calculate originality stats per source
    source_stats = df.groupby('source').agg({
        'is_original': ['sum', 'count']
    }).reset_index()
    source_stats.columns = ['source', 'original_count', 'total_count']
    source_stats['originality_rate'] = source_stats['original_count'] / source_stats['total_count']
    source_stats['copied_count'] = source_stats['total_count'] - source_stats['original_count']
    
    # Filter to sources with at least 2 articles AND that have some copies
    source_stats = source_stats[(source_stats['total_count'] >= 2) & (source_stats['copied_count'] > 0)]
    
    # Sort by copied_count descending to show biggest copiers, take top 15
    source_stats = source_stats.sort_values('copied_count', ascending=True).tail(15)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=source_stats['source'],
        x=source_stats['original_count'],
        name='Original',
        orientation='h',
        marker_color='#2ecc71'
    ))
    
    fig.add_trace(go.Bar(
        y=source_stats['source'],
        x=source_stats['copied_count'],
        name='Copied/Similar',
        orientation='h',
        marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        barmode='stack',
        title='Source Originality (Top 15)',
        xaxis_title='Number of Articles',
        yaxis_title='',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=150, r=20, t=50, b=50)
    )
    
    return fig


def render_similarity_distribution(df: pd.DataFrame):
    """Render histogram of similarity scores for duplicates/similar articles only."""
    if df is None or 'similarity_score' not in df.columns:
        return None
    
    # Filter to only non-original articles (duplicates/similar)
    # Originals have score=1.0 which skews the histogram
    if 'is_original' in df.columns:
        scores = df[(df['is_original'] == False) & (df['similarity_score'] > 0)]['similarity_score']
    else:
        # Fallback: exclude score=1.0 (originals)
        scores = df[(df['similarity_score'] > 0) & (df['similarity_score'] < 1.0)]['similarity_score']
    
    if len(scores) == 0:
        return None
    
    fig = px.histogram(
        scores,
        nbins=30,
        title='Distribution of Similarity Scores',
        labels={'value': 'Similarity Score', 'count': 'Number of Articles'},
        color_discrete_sequence=['#3498db']
    )
    
    # Add threshold lines
    fig.add_vline(x=0.85, line_dash="dash", line_color="red", 
                  annotation_text="Duplicate Threshold (0.85)")
    fig.add_vline(x=0.70, line_dash="dash", line_color="orange",
                  annotation_text="Similar Threshold (0.70)")
    
    fig.update_layout(
        height=300,
        showlegend=False,
        xaxis_title='Similarity Score',
        yaxis_title='Count'
    )
    
    return fig


def render_spread_chains(df: pd.DataFrame, top_n: int = 5):
    """Render top story spread chains."""
    if df is None or 'duplicate_of' not in df.columns:
        return
    
    # Find stories with multiple copies
    original_stories = df[df['is_original']]
    chains = []
    
    for idx in original_stories.index:
        copies = df[df['duplicate_of'] == idx]
        if len(copies) > 0:
            chains.append({
                'original_title': df.at[idx, 'title'][:80] + '...' if len(df.at[idx, 'title']) > 80 else df.at[idx, 'title'],
                'original_source': df.at[idx, 'source'],
                'original_date': df.at[idx, 'publishedAt'] if 'publishedAt' in df.columns else 'N/A',
                'copy_count': len(copies),
                'copy_sources': list(copies['source'].unique())
            })
    
    # Sort by copy count
    chains = sorted(chains, key=lambda x: x['copy_count'], reverse=True)[:top_n]
    
    if chains:
        st.subheader(f"🔗 Top {top_n} Most Copied Stories")
        
        for i, chain in enumerate(chains):
            with st.expander(f"📰 {chain['original_title']}", expanded=(i==0)):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Original Source:** {chain['original_source']}")
                    st.write(f"**Published:** {chain['original_date']}")
                with col2:
                    st.metric("Copies", chain['copy_count'])
                
                st.write("**Copied by:**")
                for source in chain['copy_sources']:
                    st.write(f"  • {source}")


def add_duplicate_analysis_to_dashboard():
    """Main function to add news spread analysis section to dashboard."""
    
    st.header("🔍 News Spread & Copy-Paste Analysis")
    st.markdown("*Tracking how news stories spread across different sources*")
    
    # Load data
    df = load_duplicate_analysis()
    
    if df is None:
        st.warning("⚠️ Spread analysis not available. Rerun preprocessing with spread detection.")
        st.code("python -c \"from src.preprocess_articles import clean_articles; clean_articles(detect_spread=True)\"")
        return
    
    # Calculate summary stats - handle different category names
    total = len(df)
    originals = len(df[df['is_original'] == True])
    
    # Count by category
    exact_copies = len(df[df['similarity_category'] == 'exact_copy'])
    semantic_copies = len(df[df['similarity_category'] == 'semantic_copy'])
    paraphrased = len(df[df['similarity_category'] == 'paraphrased'])
    
    total_copies = exact_copies + semantic_copies + paraphrased
    copy_paste_ratio = total_copies / total if total > 0 else 0
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Total Articles", total)
    col2.metric("Original", originals, 
                delta=f"{(originals/total)*100:.1f}%" if total > 0 else "N/A")
    col3.metric("Exact Copies", exact_copies,
                help="Same title, different source")
    col4.metric("Semantic Copies", semantic_copies,
                help="Different title but >85% similar content")
    col5.metric("Paraphrased", paraphrased,
                help="70-85% similar content")
    
    st.divider()
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Copy-Paste Index")
        gauge_fig = render_copy_paste_gauge(copy_paste_ratio)
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Explanation of the gauge
        st.caption(f"""
        **What does this mean?** Out of {total} articles, **{total_copies} ({copy_paste_ratio*100:.0f}%)** 
        are copies or paraphrases of other news. Lower percentage = more original journalism.
        """)
        
        # Category breakdown pie
        categories = {
            'Original': originals,
            'Exact Copy': exact_copies,
            'Semantic Copy': semantic_copies, 
            'Paraphrased': paraphrased
        }
        categories = {k: v for k, v in categories.items() if v > 0}
        
        if categories:
            fig_pie = px.pie(
                names=list(categories.keys()),
                values=list(categories.values()),
                title='Article Categories',
                color_discrete_sequence=['#2ecc71', '#e74c3c', '#e67e22', '#f1c40f']
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📈 Similarity Distribution")
        hist_fig = render_similarity_distribution(df)
        if hist_fig:
            st.plotly_chart(hist_fig, use_container_width=True)
        
        # Interpretation
        st.markdown("---")
        if copy_paste_ratio < 0.10:
            st.success("✅ **Low copy-paste ratio.** Most articles are original content.")
        elif copy_paste_ratio < 0.25:
            st.info("ℹ️ **Moderate copy-paste ratio.** Some news recycling is occurring.")
        else:
            st.warning("⚠️ **High copy-paste ratio.** Significant news recycling detected.")
    
    # Source originality chart
    st.subheader("🏆 Source Originality Ranking")
    st.markdown("*Which sources create original content vs which copy from others*")
    bar_fig = render_source_originality_chart(df)
    if bar_fig:
        st.plotly_chart(bar_fig, use_container_width=True)
    
    # Story spread chains
    render_spread_chains(df, top_n=5)
    
    # Detailed table
    with st.expander("📋 Detailed Spread Data"):
        display_cols = ['title', 'source', 'publishedAt', 'similarity_category', 'similarity_score', 'original_source']
        display_cols = [c for c in display_cols if c in df.columns]
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox(
                "Filter by category:",
                ['All', 'original', 'exact_copy', 'semantic_copy', 'paraphrased']
            )
        with col2:
            source_filter = st.selectbox(
                "Filter by source:",
                ['All'] + sorted(df['source'].unique().tolist())
            )
        
        display_df = df.copy()
        if category != 'All':
            display_df = display_df[display_df['similarity_category'] == category]
        if source_filter != 'All':
            display_df = display_df[display_df['source'] == source_filter]
        
        st.dataframe(
            display_df[display_cols].head(100),
            use_container_width=True
        )


if __name__ == "__main__":
    # Test run
    st.set_page_config(page_title="Duplicate Analysis", layout="wide")
    add_duplicate_analysis_to_dashboard()
