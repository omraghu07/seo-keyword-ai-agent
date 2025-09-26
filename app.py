# app.py
"""
Complete Keyword Research Pipeline
Integrates keyword discovery, analysis, and post-processing into one workflow
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def check_setup():
    """Check if all requirements are met"""
    print("🔍 Checking setup...")
    
    # Check API key
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("❌ SERPAPI_KEY not found in environment variables")
        print("Make sure your .env file contains: SERPAPI_KEY=your_key_here")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    # Check required packages
    required_packages = [
        ('serpapi', 'google-search-results'),
        ('pandas', 'pandas'),
        ('tabulate', 'tabulate'),
        ('openpyxl', 'openpyxl')
    ]
    
    missing = []
    for import_name, pip_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    
    if missing:
        print("❌ Missing packages:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        return False
    
    print("✅ All packages available")
    return True

def run_keyword_analysis(seed_keyword, use_volume_api=False):
    """Run the keyword analysis using the professional tool"""
    print("\n🔍 Step 1: Running keyword analysis...")
    
    try:
        # Import and run the KeywordResearchTool
        import os
        import math
        import csv
        import re
        import logging
        from datetime import date
        from typing import List, Dict, Optional, Tuple, Any
        from dataclasses import dataclass
        from serpapi import GoogleSearch

        # Configure logging to be less verbose
        logging.basicConfig(level=logging.WARNING)
        
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

        # Main analysis functions
        def find_related_keywords(seed_keyword, max_results=120):
            print(f"Finding related keywords for: '{seed_keyword}'...")
            
            params = {
                "engine": "google",
                "q": seed_keyword,
                "api_key": os.getenv("SERPAPI_KEY"),
                "hl": "en",
                "gl": "us"
            }
            
            try:
                search = GoogleSearch(params)
                results = search.get_dict()
            except Exception as e:
                print(f"Error getting related keywords: {e}")
                return []
            
            keyword_candidates = set()
            
            # Get related searches
            related_searches = results.get("related_searches", [])
            for item in related_searches:
                query = item.get("query") or item.get("suggestion")
                if query and len(query.strip()) > 0:
                    keyword_candidates.add(query.strip())
            
            # Get people also ask
            related_questions = results.get("related_questions", [])
            for item in related_questions:
                question = item.get("question") or item.get("query")
                if question and len(question.strip()) > 0:
                    keyword_candidates.add(question.strip())
            
            # Get organic titles
            organic_results = results.get("organic_results", [])
            for result in organic_results[:10]:
                title = result.get("title", "")
                if title and len(title.strip()) > 0:
                    keyword_candidates.add(title.strip())
            
            final_keywords = list(keyword_candidates)[:max_results]
            print(f"Found {len(final_keywords)} keyword candidates")
            
            return final_keywords

        def analyze_keywords(keywords, use_volume_api=False):
            print(f"Analyzing {len(keywords)} keywords...")
            
            calculator = CompetitionCalculator()
            analyzed_keywords = []
            
            for i, keyword in enumerate(keywords, 1):
                if i % 10 == 0:
                    print(f"Progress: {i}/{len(keywords)} keywords processed")
                
                # Search for keyword
                params = {
                    "engine": "google",
                    "q": keyword,
                    "api_key": os.getenv("SERPAPI_KEY"),
                    "hl": "en",
                    "gl": "us",
                    "num": 10
                }
                
                try:
                    search = GoogleSearch(params)
                    search_results = search.get_dict()
                except Exception as e:
                    print(f"Error analyzing '{keyword}': {e}")
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
            
            # Sort by opportunity score
            analyzed_keywords.sort(key=lambda x: x.opportunity_score, reverse=True)
            
            print(f"Analysis complete! {len(analyzed_keywords)} keywords analyzed")
            return analyzed_keywords

        def save_to_csv(keyword_metrics, seed_keyword, top_count=50):
            if not keyword_metrics:
                print("No data to save!")
                return None
            
            # Create filename
            today = date.today()
            safe_seed = re.sub(r"[^\w\s-]", "", seed_keyword).strip().replace(" ", "_")[:30]
            filename = f"keywords_{safe_seed}_{today}.csv"
            
            try:
                with open(filename, "w", newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    
                    # Write header
                    headers = [
                        "Keyword", "Monthly Searches", "Competition Score", 
                        "Opportunity Score", "Total Results", "Ads Count",
                        "Featured Snippet", "People Also Ask", "Knowledge Graph"
                    ]
                    writer.writerow(headers)
                    
                    # Write data
                    for metrics in keyword_metrics[:top_count]:
                        row = [
                            metrics.keyword,
                            metrics.monthly_searches,
                            metrics.competition_score,
                            metrics.opportunity_score,
                            metrics.total_results,
                            metrics.ads_count,
                            "Yes" if metrics.has_featured_snippet else "No",
                            "Yes" if metrics.has_people_also_ask else "No",
                            "Yes" if metrics.has_knowledge_graph else "No"
                        ]
                        writer.writerow(row)
                
                saved_count = min(top_count, len(keyword_metrics))
                print(f"✅ Saved {saved_count} keywords to {filename}")
                return filename
                
            except Exception as e:
                print(f"Error saving CSV: {e}")
                return None

        def display_top_results(keyword_metrics, top_count=5):
            if not keyword_metrics:
                print("No results to display!")
                return
            
            print(f"\n🏆 Top {min(top_count, len(keyword_metrics))} Keywords:")
            print("-" * 80)
            
            for i, metrics in enumerate(keyword_metrics[:top_count], 1):
                print(f"{i}. {metrics.keyword}")
                print(f"   Score: {metrics.opportunity_score} | Volume: {metrics.monthly_searches:,} | Competition: {metrics.competition_score}")
                print()

        # Run the analysis
        related_keywords = find_related_keywords(seed_keyword)
        if not related_keywords:
            print("❌ No keyword candidates found")
            return None
        
        analyzed_keywords = analyze_keywords(related_keywords, use_volume_api)
        if not analyzed_keywords:
            print("❌ No keywords analyzed successfully")
            return None
        
        filename = save_to_csv(analyzed_keywords, seed_keyword)
        display_top_results(analyzed_keywords)
        
        return filename
        
    except Exception as e:
        print(f"❌ Error in keyword analysis: {e}")
        return None

def run_postprocessing(csv_filename, seed_keyword):
    """Run post-processing on the CSV file"""
    print("\n🧹 Step 2: Running post-processing...")
    
    try:
        import pandas as pd
        import re
        import json
        from datetime import date, datetime
        
        # Try to import optional packages
        try:
            from tabulate import tabulate
            HAS_TABULATE = True
        except ImportError:
            HAS_TABULATE = False
        
        try:
            import openpyxl
            HAS_EXCEL = True
        except ImportError:
            HAS_EXCEL = False

        # Configuration
        BRAND_KEYWORDS = {
            "linkedin", "indeed", "glassdoor", "ucla", "asu", "berkeley", 
            "hennge", "ciee", "google", "facebook", "microsoft", "amazon"
        }

        def is_brand_query(keyword):
            if not keyword:
                return False
            keyword_lower = keyword.lower()
            for brand in BRAND_KEYWORDS:
                if brand in keyword_lower:
                    return True
            if re.search(r"\.(com|edu|org|net|gov|io)\b", keyword_lower):
                return True
            return False

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
            if is_brand_query(keyword):
                return "navigational"
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

        # Load and process the CSV
        print(f"Loading {csv_filename}...")
        df = pd.read_csv(csv_filename)
        print(f"Loaded {len(df)} keywords")
        
        # Clean and enhance the data
        print("Processing data...")
        
        # Standardize column names
        column_mapping = {
            'Keyword': 'Keyword',
            'Monthly Searches': 'Monthly Searches', 
            'Competition Score': 'Competition',
            'Opportunity Score': 'Opportunity Score',
            'Total Results': 'Google Results',
            'Ads Count': 'Ads Shown',
            'Featured Snippet': 'Featured Snippet?',
            'People Also Ask': 'PAA Available?',
            'Knowledge Graph': 'Knowledge Graph?'
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['Keyword'], keep='first')
        df = df.sort_values('Opportunity Score', ascending=False)
        
        # Add enhancement columns
        df['Intent'] = df['Keyword'].apply(classify_intent)
        df['Tail'] = df['Keyword'].apply(classify_tail)
        df['Is Brand/Navigational'] = df['Keyword'].apply(lambda x: "Yes" if is_brand_query(x) else "No")
        
        # Reorder columns
        column_order = [
            'Keyword', 'Intent', 'Tail', 'Is Brand/Navigational',
            'Monthly Searches', 'Competition', 'Opportunity Score',
            'Google Results', 'Ads Shown', 'Featured Snippet?',
            'PAA Available?', 'Knowledge Graph?'
        ]
        
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        
        # Create output directory
        os.makedirs("results", exist_ok=True)
        
        # Generate filenames
        today = date.today().isoformat()
        safe_seed = re.sub(r"[^\w\s-]", "", seed_keyword).strip().replace(" ", "_")[:30]
        base_name = f"keywords_{safe_seed}_{today}"
        
        csv_path = f"results/{base_name}.csv"
        excel_path = f"results/{base_name}.xlsx"
        meta_path = f"results/{base_name}.meta.json"
        
        # Save enhanced CSV
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved enhanced CSV: {csv_path}")
        
        # Save Excel if available
        if HAS_EXCEL:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                df.head(50).to_excel(writer, sheet_name="Top_50", index=False)
                df.to_excel(writer, sheet_name="All_Keywords", index=False)
            print(f"📊 Saved Excel: {excel_path}")
        
        # Save metadata
        metadata = {
            "seed_keyword": seed_keyword,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_keywords": len(df),
            "data_source": "SerpApi with heuristic search volumes",
            "methodology": "Opportunity Score = log10(volume+1) / (competition + 0.01)"
        }
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Saved metadata: {meta_path}")
        
        # Display results
        print(f"\n🏆 Top 10 Enhanced Results:")
        
        preview_df = df.head(10)
        if HAS_TABULATE:
            display_columns = ['Keyword', 'Intent', 'Tail', 'Monthly Searches', 'Competition', 'Opportunity Score']
            display_data = preview_df[display_columns]
            print(tabulate(display_data, headers="keys", tablefmt="github", showindex=False))
        else:
            for i, row in preview_df.iterrows():
                print(f"{i+1}. {row['Keyword']} | Score: {row['Opportunity Score']} | Intent: {row['Intent']} | Tail: {row['Tail']}")
        
        # Summary stats
        print(f"\n📈 Summary:")
        print(f"• Total keywords: {len(df)}")
        print(f"• Long-tail keywords: {len(df[df['Tail'] == 'long-tail'])}")
        print(f"• Non-brand keywords: {len(df[df['Is Brand/Navigational'] == 'No'])}")
        print(f"• High opportunity (score > 50): {len(df[df['Opportunity Score'] > 50])}")
        
        return csv_path, excel_path, meta_path
        
    except Exception as e:
        print(f"❌ Error in post-processing: {e}")
        return None, None, None

def run_complete_pipeline(seed_keyword, use_volume_api=False):
    """Run the complete pipeline"""
    print("🚀 Starting Complete Keyword Research Pipeline")
    print("=" * 60)
    print(f"Seed Keyword: '{seed_keyword}'")
    print("=" * 60)
    
    # Step 1: Run keyword analysis
    csv_filename = run_keyword_analysis(seed_keyword, use_volume_api)
    
    if not csv_filename:
        print("❌ Pipeline failed at Step 1")
        return False
    
    # Step 2: Run post-processing
    csv_path, excel_path, meta_path = run_postprocessing(csv_filename, seed_keyword)
    
    if not csv_path:
        print("❌ Pipeline failed at Step 2")
        return False
    
    # Final summary
    print("\n🎯 PIPELINE COMPLETE! 🎯")
    print("=" * 60)
    print(f"📁 Original CSV: {csv_filename}")
    print(f"📁 Enhanced CSV: {csv_path}")
    if excel_path:
        print(f"📁 Excel file: {excel_path}")
    if meta_path:
        print(f"📁 Metadata: {meta_path}")
    print("=" * 60)
    
    return True

def main():
    """Main function with command line support"""
    parser = argparse.ArgumentParser(description="Complete Keyword Research Pipeline")
    parser.add_argument("seed_keyword", nargs="?", default="global internship", 
                       help="Seed keyword (default: 'global internship')")
    parser.add_argument("--use-volume-api", action="store_true",
                       help="Use real volume API (requires implementation)")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check setup, don't run pipeline")
    
    args = parser.parse_args()
    
    # Check setup
    if not check_setup():
        return 1
    
    if args.check_only:
        print("✅ Setup check complete!")
        return 0
    
    # Run pipeline
    success = run_complete_pipeline(args.seed_keyword, args.use_volume_api)
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)