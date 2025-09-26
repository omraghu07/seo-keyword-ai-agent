# dashboard.py
"""
SEO Keyword Research Dashboard

A Streamlit web interface for the keyword research pipeline.
Provides interactive analysis, visualization, and download capabilities.

Requirements:
    pip install streamlit plotly pandas

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path
from datetime import date, datetime
import re
import json
import io
from typing import Optional, Tuple, Dict, Any

# Add project directories to path
project_root = Path(__file__).parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

# Import backend functions
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.error("Missing required package: python-dotenv. Install with: pip install python-dotenv")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="SEO Keyword Research Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    .stDataFrame {
        border-radius: 0.5rem;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

class KeywordDashboard:
    """Main dashboard class for SEO keyword research interface."""
    
    def __init__(self):
        """Initialize the dashboard with necessary configurations."""
        self.setup_directories()
        self.check_environment()
    
    def setup_directories(self):
        """Create necessary output directories."""
        self.output_dir = Path("output")
        self.processed_dir = self.output_dir / "processed"
        self.reports_dir = self.output_dir / "reports"
        
        self.output_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
    
    def check_environment(self):
        """Check if the environment is properly configured."""
        self.api_key = os.getenv("SERPAPI_KEY")
        self.environment_ready = bool(self.api_key)
    
    def render_header(self):
        """Render the main dashboard header."""
        st.markdown('<h1 class="main-header">🔍 SEO Keyword Research Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        if not self.environment_ready:
            st.markdown("""
            <div class="error-message">
                ⚠️ <strong>Environment Setup Required</strong><br>
                Please ensure your .env file contains: SERPAPI_KEY=your_key_here
            </div>
            """, unsafe_allow_html=True)
            return False
        
        st.markdown("""
        <div class="success-message">
            ✅ <strong>Environment Ready</strong><br>
            API key detected and ready for keyword research.
        </div>
        """, unsafe_allow_html=True)
        return True
    
    def render_sidebar(self) -> Dict[str, Any]:
        """Render the sidebar with input controls."""
        st.sidebar.markdown("## 🎯 Analysis Parameters")
        
        # Input parameters
        seed_keyword = st.sidebar.text_input(
            "🔍 Seed Keyword",
            value="global internship",
            help="Enter the main keyword to research"
        )
        
        max_candidates = st.sidebar.slider(
            "📊 Max Candidates",
            min_value=20,
            max_value=300,
            value=120,
            step=10,
            help="Maximum number of keyword candidates to analyze"
        )
        
        top_results = st.sidebar.slider(
            "🏆 Top Results",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            help="Number of top results to display and save"
        )
        
        # Advanced options
        st.sidebar.markdown("## ⚙️ Advanced Options")
        
        use_volume_api = st.sidebar.checkbox(
            "📈 Use Real Volume API",
            value=False,
            help="Enable when volume API is implemented",
            disabled=True  # Disabled until implemented
        )
        
        # Filtering options
        st.sidebar.markdown("## 🔧 Filters")
        
        min_search_volume = st.sidebar.number_input(
            "📈 Min Search Volume",
            min_value=0,
            max_value=10000,
            value=10,
            step=10,
            help="Minimum monthly search volume"
        )
        
        max_competition = st.sidebar.slider(
            "⚔️ Max Competition Score",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Maximum competition score (0=easy, 1=hard)"
        )
        
        # Run button
        run_analysis = st.sidebar.button(
            "🚀 Run Analysis",
            type="primary",
            help="Start the keyword research analysis"
        )
        
        return {
            "seed_keyword": seed_keyword,
            "max_candidates": max_candidates,
            "top_results": top_results,
            "use_volume_api": use_volume_api,
            "min_search_volume": min_search_volume,
            "max_competition": max_competition,
            "run_analysis": run_analysis
        }
    
    def run_keyword_analysis(self, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Run the keyword analysis using the backend pipeline."""
        try:
            # Import the analysis function from app.py
            sys.path.insert(0, str(project_root))
            
            # Since we need to reuse the logic from app.py, let's import what we need
            import math
            import csv
            import re
            from serpapi import GoogleSearch
            from dataclasses import dataclass
            
            @dataclass
            class KeywordMetrics:
                keyword: str
                monthly_searches: int
                competition_score: float
                opportunity_score: float
                total_results: int
                ads_count: int
                has_featured_snippet: bool
                has_people_also_ask: bool
                has_knowledge_graph: bool
            
            # Competition calculator (from your app.py)
            class CompetitionCalculator:
                WEIGHTS = {
                    'total_results': 0.50,
                    'ads': 0.25,
                    'featured_snippet': 0.15,
                    'people_also_ask': 0.07,
                    'knowledge_graph': 0.03
                }
                
                @staticmethod
                def extract_total_results(search_info):
                    if not search_info:
                        return 0
                    
                    total = (search_info.get("total_results") or 
                            search_info.get("total_results_raw") or 
                            search_info.get("total"))
                    
                    if isinstance(total, int):
                        return total
                    
                    if isinstance(total, str):
                        numbers_only = re.sub(r"[^\d]", "", total)
                        try:
                            return int(numbers_only) if numbers_only else 0
                        except ValueError:
                            return 0
                    
                    return 0
                
                def calculate_score(self, search_results):
                    search_info = search_results.get("search_information", {})
                    
                    total_results = self.extract_total_results(search_info)
                    normalized_results = min(math.log10(total_results + 1) / 7, 1.0)
                    
                    ads = search_results.get("ads_results", [])
                    ads_count = len(ads) if ads else 0
                    ads_score = min(ads_count / 3, 1.0)
                    
                    has_featured_snippet = bool(
                        search_results.get("featured_snippet") or 
                        search_results.get("answer_box")
                    )
                    
                    has_people_also_ask = bool(
                        search_results.get("related_questions") or 
                        search_results.get("people_also_ask")
                    )
                    
                    has_knowledge_graph = bool(search_results.get("knowledge_graph"))
                    
                    competition_score = (
                        self.WEIGHTS['total_results'] * normalized_results +
                        self.WEIGHTS['ads'] * ads_score +
                        self.WEIGHTS['featured_snippet'] * has_featured_snippet +
                        self.WEIGHTS['people_also_ask'] * has_people_also_ask +
                        self.WEIGHTS['knowledge_graph'] * has_knowledge_graph
                    )
                    
                    competition_score = max(0.0, min(1.0, competition_score))
                    
                    breakdown = {
                        "total_results": total_results,
                        "ads_count": ads_count,
                        "has_featured_snippet": has_featured_snippet,
                        "has_people_also_ask": has_people_also_ask,
                        "has_knowledge_graph": has_knowledge_graph
                    }
                    
                    return competition_score, breakdown
            
            def find_related_keywords(seed_keyword, max_results=120):
                progress_placeholder = st.empty()
                progress_placeholder.info(f"🔍 Finding related keywords for: '{seed_keyword}'...")
                
                search_params = {
                    "engine": "google",
                    "q": seed_keyword,
                    "api_key": self.api_key,
                    "hl": "en",
                    "gl": "us"
                }
                
                try:
                    search = GoogleSearch(search_params)
                    results = search.get_dict()
                except Exception as e:
                    progress_placeholder.error(f"❌ Error getting related keywords: {e}")
                    return []
                
                keyword_candidates = set()
                
                # Extract keywords from different sources
                related_searches = results.get("related_searches", [])
                for item in related_searches:
                    query = item.get("query") or item.get("suggestion")
                    if query and len(query.strip()) > 0:
                        keyword_candidates.add(query.strip())
                
                related_questions = results.get("related_questions", [])
                for item in related_questions:
                    question = item.get("question") or item.get("query")
                    if question and len(question.strip()) > 0:
                        keyword_candidates.add(question.strip())
                
                organic_results = results.get("organic_results", [])
                for result in organic_results[:10]:
                    title = result.get("title", "")
                    if title and len(title.strip()) > 0:
                        keyword_candidates.add(title.strip())
                
                final_keywords = list(keyword_candidates)[:max_results]
                progress_placeholder.success(f"✅ Found {len(final_keywords)} keyword candidates")
                return final_keywords
            
            def analyze_keywords_batch(keywords):
                calculator = CompetitionCalculator()
                analyzed_keywords = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, keyword in enumerate(keywords):
                    progress = (i + 1) / len(keywords)
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing keyword {i+1}/{len(keywords)}: {keyword}")
                    
                    # Search for keyword
                    search_params = {
                        "engine": "google",
                        "q": keyword,
                        "api_key": self.api_key,
                        "hl": "en",
                        "gl": "us",
                        "num": 10
                    }
                    
                    try:
                        search = GoogleSearch(search_params)
                        search_results = search.get_dict()
                    except Exception as e:
                        continue
                    
                    # Calculate competition
                    competition_score, breakdown = calculator.calculate_score(search_results)
                    
                    # Estimate volume
                    word_count = len(keyword.split())
                    search_volume = max(10, 10000 // (word_count + 1))
                    
                    # Calculate opportunity score
                    volume_score = math.log10(search_volume + 1)
                    opportunity_score = volume_score / (competition_score + 0.01)
                    
                    metrics = KeywordMetrics(
                        keyword=keyword,
                        monthly_searches=search_volume,
                        competition_score=round(competition_score, 4),
                        opportunity_score=round(opportunity_score, 2),
                        total_results=breakdown["total_results"],
                        ads_count=breakdown["ads_count"],
                        has_featured_snippet=breakdown["has_featured_snippet"],
                        has_people_also_ask=breakdown["has_people_also_ask"],
                        has_knowledge_graph=breakdown["has_knowledge_graph"]
                    )
                    
                    analyzed_keywords.append(metrics)
                
                progress_bar.empty()
                status_text.empty()
                
                # Sort by opportunity score
                analyzed_keywords.sort(key=lambda x: x.opportunity_score, reverse=True)
                return analyzed_keywords
            
            # Run the analysis
            with st.spinner("🔍 Discovering related keywords..."):
                related_keywords = find_related_keywords(
                    params["seed_keyword"], 
                    params["max_candidates"]
                )
            
            if not related_keywords:
                st.error("❌ No keyword candidates found. Please check your API key and try again.")
                return None
            
            with st.spinner("📊 Analyzing keywords and calculating scores..."):
                analyzed_keywords = analyze_keywords_batch(related_keywords)
            
            if not analyzed_keywords:
                st.error("❌ No keywords were successfully analyzed.")
                return None
            
            # Convert to DataFrame
            data = []
            for metrics in analyzed_keywords:
                data.append({
                    'Keyword': metrics.keyword,
                    'Monthly Searches': metrics.monthly_searches,
                    'Competition': metrics.competition_score,
                    'Opportunity Score': metrics.opportunity_score,
                    'Total Results': metrics.total_results,
                    'Ads Count': metrics.ads_count,
                    'Featured Snippet': 'Yes' if metrics.has_featured_snippet else 'No',
                    'People Also Ask': 'Yes' if metrics.has_people_also_ask else 'No',
                    'Knowledge Graph': 'Yes' if metrics.has_knowledge_graph else 'No'
                })
            
            df = pd.DataFrame(data)
            
            # Apply filters
            df = df[
                (df['Monthly Searches'] >= params['min_search_volume']) &
                (df['Competition'] <= params['max_competition'])
            ]
            
            return df
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            return None
    
    def add_enhancement_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add intent and tail classification columns."""
        def classify_intent(keyword):
            if not keyword:
                return "informational"
            
            k = keyword.lower()
            if any(signal in k for signal in ["how to", "what is", "why", "guide", "tutorial"]):
                return "informational"
            if any(signal in k for signal in ["buy", "price", "cost", "apply", "register"]):
                return "transactional"
            if any(signal in k for signal in ["best", "top", "compare", "vs", "reviews"]):
                return "commercial"
            return "informational"
        
        def classify_tail(keyword):
            if not keyword:
                return "short-tail"
            word_count = len(str(keyword).split())
            if word_count >= 4:
                return "long-tail"
            elif word_count == 3:
                return "mid-tail"
            else:
                return "short-tail"
        
        df['Intent'] = df['Keyword'].apply(classify_intent)
        df['Tail'] = df['Keyword'].apply(classify_tail)
        
        return df
    
    def render_summary_metrics(self, df: pd.DataFrame):
        """Render summary metrics cards."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Total Keywords</h3>
                <h2 style="color: #1f77b4;">{}</h2>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            avg_score = df['Opportunity Score'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3>⭐ Avg Opportunity Score</h3>
                <h2 style="color: #ff7f0e;">{:.2f}</h2>
            </div>
            """.format(avg_score), unsafe_allow_html=True)
        
        with col3:
            high_opportunity = len(df[df['Opportunity Score'] > 50])
            st.markdown("""
            <div class="metric-card">
                <h3>🚀 High Opportunity</h3>
                <h2 style="color: #2ca02c;">{}</h2>
            </div>
            """.format(high_opportunity), unsafe_allow_html=True)
        
        with col4:
            long_tail = len(df[df['Tail'] == 'long-tail'])
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Long-tail Keywords</h3>
                <h2 style="color: #d62728;">{}</h2>
            </div>
            """.format(long_tail), unsafe_allow_html=True)
    
    def render_top_keywords_table(self, df: pd.DataFrame, top_n: int = 10):
        """Render the top keywords table with styling."""
        st.markdown("## 🏆 Top Keyword Opportunities")
        
        if df.empty:
            st.warning("No keywords to display.")
            return
        
        # Prepare display DataFrame
        display_df = df.head(top_n).copy()
        
        # Format columns for better display
        display_df['Monthly Searches'] = display_df['Monthly Searches'].apply(lambda x: f"{x:,}")
        display_df['Total Results'] = display_df['Total Results'].apply(lambda x: f"{x:,}")
        
        # Style the dataframe
        def highlight_max_score(s):
            is_max = s == s.max()
            return ['background-color: lightgreen' if v else '' for v in is_max]
        
        styled_df = display_df.style.apply(
            highlight_max_score, 
            subset=['Opportunity Score']
        ).format({
            'Competition': '{:.3f}',
            'Opportunity Score': '{:.2f}'
        })
        
        st.dataframe(styled_df, use_container_width=True)
    
    def render_visualizations(self, df: pd.DataFrame):
        """Render interactive charts and visualizations."""
        if df.empty:
            st.warning("No data available for visualization.")
            return
        
        # Chart selection tabs
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📊 Opportunity Scores", "🎯 Intent Analysis", "💹 Volume vs Competition"])
        
        with chart_tab1:
            st.markdown("### Top 10 Keywords by Opportunity Score")
            top_10 = df.head(10)
            
            fig = px.bar(
                top_10,
                x='Opportunity Score',
                y='Keyword',
                orientation='h',
                title="Top 10 Keyword Opportunities",
                color='Opportunity Score',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_tab2:
            st.markdown("### Intent Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                intent_counts = df['Intent'].value_counts()
                fig_pie = px.pie(
                    values=intent_counts.values,
                    names=intent_counts.index,
                    title="Search Intent Distribution",
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                tail_counts = df['Tail'].value_counts()
                fig_tail = px.pie(
                    values=tail_counts.values,
                    names=tail_counts.index,
                    title="Keyword Tail Distribution",
                    color_discrete_sequence=['#9467bd', '#8c564b', '#e377c2']
                )
                st.plotly_chart(fig_tail, use_container_width=True)
        
        with chart_tab3:
            st.markdown("### Search Volume vs Competition Analysis")
            
            fig_scatter = px.scatter(
                df.head(50),  # Limit to top 50 for readability
                x='Competition',
                y='Monthly Searches',
                size='Opportunity Score',
                color='Intent',
                hover_name='Keyword',
                title="Search Volume vs Competition (Size = Opportunity Score)",
                labels={'Competition': 'Competition Score', 'Monthly Searches': 'Est. Monthly Searches'}
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    def save_results(self, df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[str, str, str]:
        """Save results to files and return file paths."""
        if df.empty:
            return None, None, None
        
        # Generate file names
        today = date.today().isoformat()
        safe_seed = re.sub(r"[^\w\s-]", "", params['seed_keyword']).strip().replace(" ", "_")[:30]
        base_name = f"keywords_{safe_seed}_{today}"
        
        # File paths
        csv_path = self.processed_dir / f"{base_name}.csv"
        excel_path = self.processed_dir / f"{base_name}.xlsx"
        report_path = self.reports_dir / f"{base_name}_report.json"
        
        try:
            # Save CSV
            df.to_csv(csv_path, index=False)
            
            # Save Excel with multiple sheets
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.head(params['top_results']).to_excel(writer, sheet_name='Top_Results', index=False)
                df.to_excel(writer, sheet_name='All_Keywords', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': [
                        'Total Keywords',
                        'Average Opportunity Score',
                        'High Opportunity Keywords (>50)',
                        'Long-tail Keywords',
                        'Informational Intent',
                        'Commercial Intent',
                        'Transactional Intent'
                    ],
                    'Value': [
                        len(df),
                        round(df['Opportunity Score'].mean(), 2),
                        len(df[df['Opportunity Score'] > 50]),
                        len(df[df['Tail'] == 'long-tail']),
                        len(df[df['Intent'] == 'informational']),
                        len(df[df['Intent'] == 'commercial']),
                        len(df[df['Intent'] == 'transactional'])
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Save JSON report
            report_data = {
                'analysis_date': datetime.now().isoformat(),
                'seed_keyword': params['seed_keyword'],
                'parameters': {
                    'max_candidates': params['max_candidates'],
                    'top_results': params['top_results'],
                    'min_search_volume': params['min_search_volume'],
                    'max_competition': params['max_competition']
                },
                'summary': {
                    'total_keywords': len(df),
                    'average_opportunity_score': float(df['Opportunity Score'].mean()),
                    'top_keyword': df.iloc[0]['Keyword'] if not df.empty else None,
                    'intent_distribution': df['Intent'].value_counts().to_dict(),
                    'tail_distribution': df['Tail'].value_counts().to_dict()
                }
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            return str(csv_path), str(excel_path), str(report_path)
            
        except Exception as e:
            st.error(f"❌ Error saving files: {e}")
            return None, None, None
    
    def render_download_section(self, csv_path: str, excel_path: str, report_path: str):
        """Render download buttons for generated files."""
        st.markdown("## 📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        if csv_path and os.path.exists(csv_path):
            with col1:
                with open(csv_path, 'rb') as file:
                    st.download_button(
                        label="📊 Download CSV",
                        data=file.read(),
                        file_name=os.path.basename(csv_path),
                        mime="text/csv"
                    )
        
        if excel_path and os.path.exists(excel_path):
            with col2:
                with open(excel_path, 'rb') as file:
                    st.download_button(
                        label="📈 Download Excel",
                        data=file.read(),
                        file_name=os.path.basename(excel_path),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        if report_path and os.path.exists(report_path):
            with col3:
                with open(report_path, 'rb') as file:
                    st.download_button(
                        label="📋 Download Report",
                        data=file.read(),
                        file_name=os.path.basename(report_path),
                        mime="application/json"
                    )
    
    def run(self):
        """Main dashboard execution method."""
        # Render header
        if not self.render_header():
            st.stop()
        
        # Render sidebar
        params = self.render_sidebar()
        
        # Main content area
        if params["run_analysis"]:
            # Store analysis state
            if 'analysis_complete' not in st.session_state:
                st.session_state.analysis_complete = False
            
            # Run analysis
            df = self.run_keyword_analysis(params)
            
            if df is not None and not df.empty:
                # Add enhancement columns
                df = self.add_enhancement_columns(df)
                
                # Store results in session state
                st.session_state.results_df = df
                st.session_state.analysis_params = params
                st.session_state.analysis_complete = True
                
                # Success message
                st.success(f"✅ Analysis complete! Found {len(df)} keywords matching your criteria.")
        
        # Display results if analysis is complete
        if st.session_state.get('analysis_complete', False) and 'results_df' in st.session_state:
            df = st.session_state.results_df
            params = st.session_state.analysis_params
            
            # Render summary metrics
            self.render_summary_metrics(df)
            
            # Create view toggle
            view_option = st.radio("📋 Choose View", ["Table View", "Chart View"], horizontal=True)
            
            if view_option == "Table View":
                self.render_top_keywords_table(df, params['top_results'])
            else:
                self.render_visualizations(df)
            
            # Save results and provide downloads
            with st.spinner("💾 Preparing download files..."):
                csv_path, excel_path, report_path = self.save_results(df, params)
            
            if csv_path:
                self.render_download_section(csv_path, excel_path, report_path)
        
        elif not st.session_state.get('analysis_complete', False):
            # Show welcome message
            st.markdown("""
            ## 👋 Welcome to the SEO Keyword Research Dashboard
            
            This dashboard helps you discover and analyze keyword opportunities using advanced SEO metrics.
            
            ### 🚀 Getting Started:
            1. **Enter your seed keyword** in the sidebar (e.g., "digital marketing")
            2. **Adjust analysis parameters** (candidates, results, filters)
            3. **Click "Run Analysis"** to start the keyword research
            4. **Explore results** through tables and interactive charts
            5. **Download reports** in CSV, Excel, or JSON format
            
            ### 📊 Features:
            - **Real-time keyword discovery** using SerpAPI
            - **Competition analysis** based on SERP features
            - **Intent classification** (informational, commercial, transactional)
            - **Interactive visualizations** with Plotly charts
            - **Advanced filtering** by volume and competition
            - **Multi-format exports** (CSV, Excel, JSON reports)
            """)


def main():
    """Main function to run the Streamlit dashboard."""
    dashboard = KeywordDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()