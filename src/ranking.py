"""
Professional Keyword Research Tool

A comprehensive tool for analyzing keyword opportunities using SerpApi.
Calculates competition scores and opportunity rankings based on SERP analysis.

Requirements:
    pip install serpapi tabulate python-dotenv

Setup:
    1. Create a .env file with your SerpApi key: SERPAPI_KEY=your_key_here
    2. Run the script with your desired seed keyword
"""

import os
import math
import csv
import re
import logging
from datetime import date
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from serpapi import GoogleSearch

# Optional dependency for better table formatting
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("💡 Tip: Install 'tabulate' for prettier output: pip install tabulate")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class KeywordMetrics:
    """Container for keyword analysis results."""
    keyword: str
    monthly_searches: int
    competition_score: float
    opportunity_score: float
    total_results: int
    ads_count: int
    has_featured_snippet: bool
    has_people_also_ask: bool
    has_knowledge_graph: bool


class Config:
    """Configuration settings for the keyword research tool."""
    
    def __init__(self):
        load_dotenv()
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.default_location = "United States"
        self.results_per_query = 10
        self.max_related_keywords = 150
        self.top_keywords_to_save = 50
        self.progress_update_interval = 10
        
        if not self.serpapi_key:
            raise ValueError("SERPAPI_KEY not found in environment variables")


class CompetitionCalculator:
    """Calculates keyword competition scores based on SERP features."""
    
    # Scoring weights for different competition factors
    WEIGHTS = {
        'total_results': 0.50,
        'ads': 0.25,
        'featured_snippet': 0.15,
        'people_also_ask': 0.07,
        'knowledge_graph': 0.03
    }
    
    @staticmethod
    def extract_total_results(search_info: Dict[str, Any]) -> int:
        """
        Extract total results count from SerpApi response.
        
        Args:
            search_info: Search information dictionary from SerpApi
            
        Returns:
            Total number of results as integer, 0 if not found
        """
        if not search_info:
            return 0
        
        # Try different possible field names
        total = (search_info.get("total_results") or 
                search_info.get("total_results_raw") or 
                search_info.get("total"))
        
        if isinstance(total, int):
            return total
        
        if isinstance(total, str):
            # Extract only digits (remove commas, spaces, etc.)
            numbers_only = re.sub(r"[^\d]", "", total)
            try:
                return int(numbers_only) if numbers_only else 0
            except ValueError:
                return 0
        
        return 0
    
    def calculate_score(self, search_results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate competition score based on SERP features.
        
        Args:
            search_results: Complete search results from SerpApi
            
        Returns:
            Tuple of (competition_score, analysis_breakdown)
            Score ranges from 0-1 where 1 = very competitive
        """
        search_info = search_results.get("search_information", {})
        
        # Factor 1: Total number of results (normalized using log scale)
        total_results = self.extract_total_results(search_info)
        normalized_results = min(math.log10(total_results + 1) / 7, 1.0)
        
        # Factor 2: Number of ads (more ads = more competition)
        ads = search_results.get("ads_results", [])
        ads_count = len(ads) if ads else 0
        ads_score = min(ads_count / 3, 1.0)
        
        # Factor 3: SERP features that make ranking more difficult
        has_featured_snippet = bool(
            search_results.get("featured_snippet") or 
            search_results.get("answer_box")
        )
        
        has_people_also_ask = bool(
            search_results.get("related_questions") or 
            search_results.get("people_also_ask")
        )
        
        has_knowledge_graph = bool(search_results.get("knowledge_graph"))
        
        # Calculate weighted competition score
        competition_score = (
            self.WEIGHTS['total_results'] * normalized_results +
            self.WEIGHTS['ads'] * ads_score +
            self.WEIGHTS['featured_snippet'] * has_featured_snippet +
            self.WEIGHTS['people_also_ask'] * has_people_also_ask +
            self.WEIGHTS['knowledge_graph'] * has_knowledge_graph
        )
        
        # Ensure score stays within bounds
        competition_score = max(0.0, min(1.0, competition_score))
        
        # Create analysis breakdown for reporting
        breakdown = {
            "total_results": total_results,
            "ads_count": ads_count,
            "has_featured_snippet": has_featured_snippet,
            "has_people_also_ask": has_people_also_ask,
            "has_knowledge_graph": has_knowledge_graph
        }
        
        return competition_score, breakdown


class SearchVolumeEstimator:
    """Handles search volume estimation and integration with volume APIs."""
    
    def get_search_volume(self, keyword: str) -> Optional[int]:
        """
        Get search volume for a keyword.
        
        TODO: Integrate with DataForSEO, Google Keyword Planner, or similar API
        
        Args:
            keyword: The keyword to get volume for
            
        Returns:
            Monthly search volume or None if unavailable
        """
        # Placeholder for real volume API integration
        # Examples of what you might implement:
        # - return self._call_dataforseo_api(keyword)
        # - return self._call_google_ads_api(keyword)
        return None
    
    def estimate_volume(self, keyword: str) -> int:
        """
        Estimate search volume using simple heuristics.
        
        Args:
            keyword: The keyword to estimate volume for
            
        Returns:
            Estimated monthly search volume
        """
        # Simple heuristic: longer phrases typically have lower volume
        word_count = len(keyword.split())
        # This is rough estimation - replace with real data when possible
        return max(10, 10000 // (word_count + 1))


class KeywordDiscovery:
    """Discovers related keywords from search results."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def find_related_keywords(self, seed_keyword: str) -> List[str]:
        """
        Find related keywords from Google's suggestions and related searches.
        
        Args:
            seed_keyword: The base keyword to find related terms for
            
        Returns:
            List of related keyword candidates
        """
        logger.info(f"Discovering related keywords for: '{seed_keyword}'")
        
        search_params = {
            "engine": "google",
            "q": seed_keyword,
            "api_key": self.config.serpapi_key,
            "hl": "en",
            "gl": "us"
        }
        
        try:
            search = GoogleSearch(search_params)
            results = search.get_dict()
        except Exception as e:
            logger.error(f"Failed to get related keywords: {e}")
            return []
        
        keyword_candidates = set()
        
        # Extract keywords from different sources
        self._extract_from_related_searches(results, keyword_candidates)
        self._extract_from_people_also_ask(results, keyword_candidates)
        self._extract_from_organic_titles(results, keyword_candidates)
        
        # Convert to list and limit results
        final_keywords = list(keyword_candidates)[:self.config.max_related_keywords]
        logger.info(f"Found {len(final_keywords)} keyword candidates")
        
        return final_keywords
    
    def _extract_from_related_searches(self, results: Dict[str, Any], 
                                     candidates: set) -> None:
        """Extract keywords from 'related searches' section."""
        related_searches = results.get("related_searches", [])
        for item in related_searches:
            query = item.get("query") or item.get("suggestion")
            if query and len(query.strip()) > 0:
                candidates.add(query.strip())
    
    def _extract_from_people_also_ask(self, results: Dict[str, Any], 
                                    candidates: set) -> None:
        """Extract keywords from 'People also ask' questions."""
        related_questions = results.get("related_questions", [])
        for item in related_questions:
            question = item.get("question") or item.get("query")
            if question and len(question.strip()) > 0:
                candidates.add(question.strip())
    
    def _extract_from_organic_titles(self, results: Dict[str, Any], 
                                   candidates: set) -> None:
        """Extract potential keywords from organic result titles."""
        organic_results = results.get("organic_results", [])
        for result in organic_results[:10]:  # Only top 10 results
            title = result.get("title", "")
            if title and len(title.strip()) > 0:
                candidates.add(title.strip())


class KeywordAnalyzer:
    """Main class for analyzing keywords and calculating opportunity scores."""
    
    def __init__(self, config: Config):
        self.config = config
        self.competition_calc = CompetitionCalculator()
        self.volume_estimator = SearchVolumeEstimator()
        self.keyword_discovery = KeywordDiscovery(config)
    
    def search_google(self, keyword: str) -> Dict[str, Any]:
        """
        Fetch search results for a keyword using SerpApi.
        
        Args:
            keyword: The keyword to search for
            
        Returns:
            Search results dictionary from SerpApi
        """
        search_params = {
            "engine": "google",
            "q": keyword,
            "api_key": self.config.serpapi_key,
            "hl": "en",
            "gl": "us",
            "num": self.config.results_per_query
        }
        
        try:
            search = GoogleSearch(search_params)
            return search.get_dict()
        except Exception as e:
            logger.error(f"Search failed for '{keyword}': {e}")
            return {}
    
    def analyze_keyword(self, keyword: str, use_volume_api: bool = False) -> Optional[KeywordMetrics]:
        """
        Analyze a single keyword and calculate its opportunity score.
        
        Args:
            keyword: The keyword to analyze
            use_volume_api: Whether to use real volume API (not implemented yet)
            
        Returns:
            KeywordMetrics object or None if analysis failed
        """
        # Get search results
        search_results = self.search_google(keyword)
        if not search_results:
            return None
        
        # Calculate competition score
        competition_score, breakdown = self.competition_calc.calculate_score(search_results)
        
        # Get or estimate search volume
        if use_volume_api:
            search_volume = self.volume_estimator.get_search_volume(keyword)
        else:
            search_volume = None
            
        if search_volume is None:
            search_volume = self.volume_estimator.estimate_volume(keyword)
        
        # Calculate opportunity score
        # Higher volume = better, lower competition = better
        volume_score = math.log10(search_volume + 1)
        opportunity_score = volume_score / (competition_score + 0.01)  # Avoid division by zero
        
        return KeywordMetrics(
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
    
    def analyze_keywords_batch(self, keywords: List[str], 
                             use_volume_api: bool = False) -> List[KeywordMetrics]:
        """
        Analyze multiple keywords and return sorted results.
        
        Args:
            keywords: List of keywords to analyze
            use_volume_api: Whether to use real volume API
            
        Returns:
            List of KeywordMetrics sorted by opportunity score (highest first)
        """
        logger.info(f"Analyzing {len(keywords)} keywords...")
        analyzed_keywords = []
        
        for i, keyword in enumerate(keywords, 1):
            if i % self.config.progress_update_interval == 0:
                logger.info(f"Progress: {i}/{len(keywords)} keywords processed")
            
            metrics = self.analyze_keyword(keyword, use_volume_api)
            if metrics:
                analyzed_keywords.append(metrics)
        
        # Sort by opportunity score (highest first)
        analyzed_keywords.sort(key=lambda x: x.opportunity_score, reverse=True)
        
        logger.info(f"Analysis complete! {len(analyzed_keywords)} keywords analyzed")
        return analyzed_keywords


class ResultsExporter:
    """Handles exporting results to various formats."""
    
    def save_to_csv(self, keyword_metrics: List[KeywordMetrics], 
                    base_filename: str = "keyword_analysis", 
                    top_count: int = 50) -> Optional[str]:
        """
        Save keyword analysis results to CSV file.
        
        Args:
            keyword_metrics: List of analyzed keyword metrics
            base_filename: Base name for the output file
            top_count: Number of top results to save
            
        Returns:
            Filename if successful, None if failed
        """
        if not keyword_metrics:
            logger.warning("No data to save!")
            return None
        
        # Create filename with timestamp
        today = date.today()
        filename = f"{base_filename}_{today}.csv"
        
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
                
                # Write data rows
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
            logger.info(f"✅ Results saved to {filename} ({saved_count} keywords)")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            return None
    
    def display_top_results(self, keyword_metrics: List[KeywordMetrics], 
                          top_count: int = 5) -> None:
        """
        Display top results in formatted table.
        
        Args:
            keyword_metrics: List of analyzed keyword metrics
            top_count: Number of top results to display
        """
        if not keyword_metrics:
            logger.warning("No results to display!")
            return
        
        top_results = keyword_metrics[:top_count]
        
        print(f"\n🏆 Top {len(top_results)} Keyword Opportunities:")
        
        if HAS_TABULATE:
            # Create table data
            table_data = []
            for metrics in top_results:
                table_data.append([
                    metrics.keyword,
                    f"{metrics.monthly_searches:,}",
                    f"{metrics.competition_score:.3f}",
                    f"{metrics.opportunity_score:.2f}",
                    f"{metrics.total_results:,}",
                    metrics.ads_count
                ])
            
            headers = ["Keyword", "Volume", "Competition", "Score", "Results", "Ads"]
            print(tabulate(table_data, headers=headers, tablefmt="pretty"))
        else:
            # Fallback to simple format
            for i, metrics in enumerate(top_results, 1):
                print(f"{i}. {metrics.keyword}")
                print(f"   Score: {metrics.opportunity_score}, "
                      f"Volume: {metrics.monthly_searches:,}, "
                      f"Competition: {metrics.competition_score:.3f}")


class KeywordResearchTool:
    """Main application class that orchestrates the keyword research process."""
    
    def __init__(self, seed_keyword: str):
        self.seed_keyword = seed_keyword
        self.config = Config()
        self.analyzer = KeywordAnalyzer(self.config)
        self.exporter = ResultsExporter()
    
    def run_analysis(self, use_volume_api: bool = False) -> None:
        """
        Run the complete keyword research analysis.
        
        Args:
            use_volume_api: Whether to use real volume API (requires implementation)
        """
        print("🔍 Starting keyword research analysis...")
        print(f"Seed keyword: '{self.seed_keyword}'")
        
        try:
            # Step 1: Discover related keywords
            related_keywords = self.analyzer.keyword_discovery.find_related_keywords(
                self.seed_keyword
            )
            
            if not related_keywords:
                logger.error("No keyword candidates found. Check your SerpApi key.")
                return
            
            # Step 2: Analyze keywords and calculate scores
            analyzed_keywords = self.analyzer.analyze_keywords_batch(
                related_keywords, use_volume_api
            )
            
            if not analyzed_keywords:
                logger.error("No keywords were successfully analyzed.")
                return
            
            # Step 3: Save results to file
            self.exporter.save_to_csv(
                analyzed_keywords, 
                base_filename=f"keywords_{self.seed_keyword.replace(' ', '_')}",
                top_count=self.config.top_keywords_to_save
            )
            
            # Step 4: Display top results
            self.exporter.display_top_results(analyzed_keywords, top_count=5)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


def main():
    """Main entry point for the keyword research tool."""
    # Configuration
    SEED_KEYWORD = "global internship"
    USE_VOLUME_API = False  # Set to True when you implement get_search_volume()
    
    try:
        tool = KeywordResearchTool(SEED_KEYWORD)
        tool.run_analysis(use_volume_api=USE_VOLUME_API)
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print("\n💡 Setup Instructions:")
        print("1. Create a .env file in the same directory")
        print("2. Add your SerpApi key: SERPAPI_KEY=your_key_here")
        print("3. Get your free key at: https://serpapi.com/")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()