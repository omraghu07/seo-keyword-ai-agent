"""
FastAPI server for SEO keyword research API - Optimized for First Page Rankings
Focus: Find 50 keywords with lowest competition and highest volume for first page potential
"""

import os
import logging
import time
import math
import re
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Query, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from serpapi import GoogleSearch
    HAS_SERPAPI = True
except ImportError:
    HAS_SERPAPI = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SEO Keyword Research API - First Page Optimizer",
    description="REST API for finding low-competition, high-volume keywords for first page rankings",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Authentication setup
API_AUTH_KEY = os.getenv("API_AUTH_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Rate limiting setup
REQUEST_TIMES = {}
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX_REQUESTS = 20

# Cache for recent requests
REQUEST_CACHE = {}
CACHE_TTL = 600  # 10 minutes for better reuse

# Thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=5)

# Competition thresholds for first page potential
FIRST_PAGE_THRESHOLDS = {
    "max_total_results": 5000000,  # Maximum 5M results for realistic competition
    "max_ads": 2,  # Fewer ads = less commercial competition
    "max_domain_authority": 50,  # Target keywords where top results aren't all high DA sites
    "min_opportunity_score": 5.0  # Minimum opportunity score for consideration
}

# Request models
class KeywordRequest(BaseModel):
    seed: str = Field(..., description="Seed keyword for research")
    top: int = Field(50, description="Number of results to return", ge=10, le=100)
    max_candidates: int = Field(100, description="Max candidates to analyze", ge=50, le=200)
    focus: str = Field("first_page", description="Optimization focus: first_page, volume, or balanced")

class KeywordResponse(BaseModel):
    seed: str = Field(..., description="The input seed keyword")
    returned: int = Field(..., description="Number of results returned")
    results: List[Dict[str, Any]] = Field(..., description="List of keyword analysis results")
    processing_time_seconds: float = Field(..., description="Time taken to process the request")
    cached: bool = Field(False, description="Whether results were served from cache")
    first_page_potential: int = Field(..., description="Number of keywords with first page potential")

def extract_total_results(search_info: Dict[str, Any]) -> int:
    """Extract total results count from search response."""
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

def analyze_serp_difficulty(search_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze SERP difficulty for first page ranking potential.
    Returns detailed competition metrics.
    """
    organic_results = search_results.get("organic_results", [])
    
    # Analyze top 10 results
    top_10_metrics = {
        "avg_title_length": 0,
        "exact_match_titles": 0,
        "high_da_sites": 0,
        "forum_sites": 0,
        "user_generated_content": 0,
        "commercial_sites": 0
    }
    
    high_da_domains = ["wikipedia", "amazon", "youtube", "facebook", "linkedin", 
                       "twitter", "instagram", "microsoft", "apple", "google"]
    forum_indicators = ["forum", "reddit", "quora", "stackexchange", "answers"]
    ugc_indicators = ["blog", "medium", "wordpress", "tumblr", "blogspot"]
    
    for i, result in enumerate(organic_results[:10]):
        link = result.get("link", "").lower()
        title = result.get("title", "").lower()
        
        # Check for high DA sites
        if any(domain in link for domain in high_da_domains):
            top_10_metrics["high_da_sites"] += 1
        
        # Check for forums (easier to outrank)
        if any(forum in link for forum in forum_indicators):
            top_10_metrics["forum_sites"] += 1
        
        # Check for UGC sites (easier to outrank)
        if any(ugc in link for ugc in ugc_indicators):
            top_10_metrics["user_generated_content"] += 1
        
        # Check for commercial intent
        if any(term in link for term in ["shop", "buy", "store", "cart"]):
            top_10_metrics["commercial_sites"] += 1
        
        # Title analysis
        top_10_metrics["avg_title_length"] += len(title.split())
    
    if organic_results:
        top_10_metrics["avg_title_length"] /= min(10, len(organic_results))
    
    return top_10_metrics

def calculate_advanced_competition_score(search_results: Dict[str, Any]) -> Tuple[float, Dict[str, Any], bool]:
    """
    Calculate advanced competition score with first page ranking potential.
    Returns (score, breakdown, first_page_potential)
    """
    search_info = search_results.get("search_information", {})
    
    # Basic metrics
    total_results = extract_total_results(search_info)
    normalized_results = min(math.log10(total_results + 1) / 7, 1.0) if total_results > 0 else 0
    
    # Ad competition
    ads = search_results.get("ads_results", [])
    ads_count = len(ads) if ads else 0
    ads_score = min(ads_count / 4, 1.0)  # Normalized to 4 ads max
    
    # SERP features that make ranking harder
    has_featured_snippet = bool(search_results.get("featured_snippet") or search_results.get("answer_box"))
    has_people_also_ask = bool(search_results.get("related_questions") or search_results.get("people_also_ask"))
    has_knowledge_graph = bool(search_results.get("knowledge_graph"))
    has_local_pack = bool(search_results.get("local_results"))
    has_shopping_results = bool(search_results.get("shopping_results"))
    
    # Analyze SERP difficulty
    serp_metrics = analyze_serp_difficulty(search_results)
    
    # Calculate competition score (0-1, lower is better)
    competition_score = (
        0.30 * normalized_results +  # Total results impact
        0.20 * ads_score +  # Ad competition
        0.10 * (1 if has_featured_snippet else 0) +
        0.10 * (1 if has_knowledge_graph else 0) +
        0.05 * (1 if has_people_also_ask else 0) +
        0.10 * (1 if has_local_pack else 0) +
        0.05 * (1 if has_shopping_results else 0) +
        0.10 * (serp_metrics["high_da_sites"] / 10)  # High DA site presence
    )
    
    competition_score = max(0.0, min(1.0, competition_score))
    
    # Opportunity indicators (things that make ranking easier)
    opportunity_factors = (
        serp_metrics["forum_sites"] / 10 * 0.3 +  # Forums are easier to outrank
        serp_metrics["user_generated_content"] / 10 * 0.2 +  # UGC is easier to outrank
        (1 if serp_metrics["avg_title_length"] > 7 else 0) * 0.1  # Long titles = less optimized
    )
    
    # Adjust competition score based on opportunities
    adjusted_competition = max(0, competition_score - opportunity_factors)
    
    # Determine first page potential
    first_page_potential = (
        total_results < FIRST_PAGE_THRESHOLDS["max_total_results"] and
        ads_count <= FIRST_PAGE_THRESHOLDS["max_ads"] and
        serp_metrics["high_da_sites"] < 5 and
        adjusted_competition < 0.6
    )
    
    breakdown = {
        "total_results": total_results,
        "ads_count": ads_count,
        "has_featured_snippet": has_featured_snippet,
        "has_people_also_ask": has_people_also_ask,
        "has_knowledge_graph": has_knowledge_graph,
        "has_local_pack": has_local_pack,
        "has_shopping_results": has_shopping_results,
        "high_da_sites": serp_metrics["high_da_sites"],
        "forum_sites": serp_metrics["forum_sites"],
        "ugc_sites": serp_metrics["user_generated_content"],
        "first_page_potential": first_page_potential
    }
    
    return adjusted_competition, breakdown, first_page_potential

def estimate_search_volume(keyword: str, search_results: Dict[str, Any]) -> int:
    """
    Estimate search volume based on various factors.
    More accurate estimation using SERP signals.
    """
    # Base estimation from keyword characteristics
    word_count = len(keyword.split())
    
    # Base volumes by word count
    base_volumes = {
        1: 10000,  # Single words tend to have high volume
        2: 5000,   # Two-word phrases
        3: 2000,   # Three-word phrases (long-tail)
        4: 1000,   # Four-word phrases
    }
    
    base_volume = base_volumes.get(word_count, 500)
    
    # Adjust based on SERP signals
    ads_count = len(search_results.get("ads_results", []))
    has_shopping = bool(search_results.get("shopping_results"))
    related_searches = len(search_results.get("related_searches", []))
    
    # More ads = higher commercial value = likely higher volume
    volume_multiplier = 1.0
    if ads_count > 3:
        volume_multiplier *= 1.5
    elif ads_count > 0:
        volume_multiplier *= 1.2
    
    if has_shopping:
        volume_multiplier *= 1.3
    
    # More related searches = popular topic
    if related_searches > 5:
        volume_multiplier *= 1.2
    
    estimated_volume = int(base_volume * volume_multiplier)
    
    # Cap at reasonable limits
    return min(max(estimated_volume, 10), 100000)

def collect_candidates_comprehensive(seed: str, max_candidates: int = 100) -> List[str]:
    """
    Comprehensive candidate collection for finding 50 low-competition keywords.
    Uses multiple strategies to find long-tail, low-competition variations.
    """
    candidates = set()
    candidates.add(seed)  # Always include the seed
    
    # Strategy 1: Question-based keywords (typically lower competition)
    question_prefixes = ["how to", "what is", "why does", "when to", "where to", 
                        "can i", "should i", "is it", "does", "will"]
    for prefix in question_prefixes:
        candidates.add(f"{prefix} {seed}")
        candidates.add(f"{prefix} {seed} work")
    
    # Strategy 2: Comparison keywords (often lower competition)
    candidates.add(f"{seed} vs")
    candidates.add(f"{seed} alternatives")
    candidates.add(f"{seed} comparison")
    candidates.add(f"best {seed} alternatives")
    
    # Strategy 3: Long-tail modifiers (lower competition)
    modifiers = {
        "informational": ["guide", "tutorial", "explained", "definition", "meaning", 
                         "examples", "tips", "tricks", "ideas", "strategies"],
        "commercial": ["review", "best", "top", "cheap", "affordable", "budget",
                      "premium", "professional", "discount", "deals"],
        "local": ["near me", "online", "services", "companies", "providers"],
        "temporal": ["2024", "2025", "latest", "new", "updated", "modern"],
        "user_intent": ["for beginners", "for students", "for professionals",
                       "for small business", "step by step", "DIY", "free"]
    }
    
    for category, mods in modifiers.items():
        for mod in mods:
            candidates.add(f"{seed} {mod}")
            candidates.add(f"{mod} {seed}")
            if category == "user_intent":
                candidates.add(f"{seed} {mod}")
    
    # Strategy 4: Problem-solving keywords (low competition gold)
    problem_terms = ["fix", "solve", "troubleshoot", "repair", "improve", "optimize"]
    for term in problem_terms:
        candidates.add(f"{term} {seed}")
        candidates.add(f"how to {term} {seed}")
    
    # Strategy 5: Use SerpAPI to find related searches and PAA questions
    if HAS_SERPAPI and SERPAPI_KEY:
        try:
            search_params = {
                "engine": "google",
                "q": seed,
                "api_key": SERPAPI_KEY,
                "hl": "en",
                "gl": "us",
                "num": 10
            }
            
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            # Get related searches (usually lower competition variations)
            related_searches = results.get("related_searches", [])[:15]
            for item in related_searches:
                query = item.get("query", "")
                if query and len(query.split()) <= 6:  # Focus on manageable length
                    candidates.add(query.lower().strip())
            
            # Get People Also Ask questions (excellent for low competition)
            related_questions = results.get("related_questions", [])[:10]
            for item in related_questions:
                question = item.get("question", "")
                if question:
                    candidates.add(question.lower().strip())
            
            # Get autocomplete suggestions (if available)
            if "google_autocomplete" in results:
                for suggestion in results["google_autocomplete"][:10]:
                    candidates.add(suggestion.lower().strip())
                    
        except Exception as e:
            logger.warning(f"SerpAPI collection failed: {e}")
    
    # Strategy 6: Industry-specific variations
    industries = ["software", "tools", "services", "products", "solutions"]
    for industry in industries:
        candidates.add(f"{seed} {industry}")
    
    # Remove duplicates and limit
    candidates = list(candidates)[:max_candidates]
    
    logger.info(f"Collected {len(candidates)} candidates for '{seed}'")
    return candidates

def score_keyword_for_first_page(keyword: str) -> Optional[Dict[str, Any]]:
    """
    Score a keyword specifically for first page ranking potential.
    Focus on finding low-competition, reasonable-volume keywords.
    """
    try:
        if not HAS_SERPAPI or not SERPAPI_KEY:
            logger.warning("SerpAPI not available, using mock data")
            return mock_keyword_score(keyword)
        
        # Search with SerpAPI
        search_params = {
            "engine": "google",
            "q": keyword,
            "api_key": SERPAPI_KEY,
            "hl": "en",
            "gl": "us",
            "num": 10  # Get top 10 results for analysis
        }
        
        search = GoogleSearch(search_params)
        search_results = search.get_dict()
        
        # Calculate competition and opportunity
        competition_score, breakdown, first_page_potential = calculate_advanced_competition_score(search_results)
        
        # Estimate search volume
        estimated_volume = estimate_search_volume(keyword, search_results)
        
        # Calculate opportunity score (higher is better)
        # Formula: (Volume ^ 0.7) / (Competition + 0.1)
        # Using 0.7 power to reduce impact of very high volumes
        volume_score = math.pow(estimated_volume, 0.7)
        opportunity_score = volume_score / (competition_score + 0.1)
        
        # Keyword difficulty classification
        if competition_score < 0.3:
            difficulty = "Easy"
        elif competition_score < 0.5:
            difficulty = "Medium"
        elif competition_score < 0.7:
            difficulty = "Hard"
        else:
            difficulty = "Very Hard"
        
        # First page ranking chance
        if first_page_potential and competition_score < 0.4:
            ranking_chance = "High"
        elif first_page_potential and competition_score < 0.6:
            ranking_chance = "Medium"
        elif competition_score < 0.7:
            ranking_chance = "Low"
        else:
            ranking_chance = "Very Low"
        
        return {
            "keyword": keyword,
            "monthly_searches": estimated_volume,
            "competition_score": round(competition_score, 4),
            "opportunity_score": round(opportunity_score, 2),
            "difficulty": difficulty,
            "first_page_potential": first_page_potential,
            "ranking_chance": ranking_chance,
            "total_results": breakdown["total_results"],
            "ads_count": breakdown["ads_count"],
            "high_da_competitors": breakdown["high_da_sites"],
            "easy_targets": breakdown["forum_sites"] + breakdown["ugc_sites"],
            "featured_snippet": "Yes" if breakdown["has_featured_snippet"] else "No",
            "people_also_ask": "Yes" if breakdown["has_people_also_ask"] else "No",
            "knowledge_graph": "Yes" if breakdown["has_knowledge_graph"] else "No",
            "local_pack": "Yes" if breakdown.get("has_local_pack") else "No"
        }
        
    except Exception as e:
        logger.error(f"Scoring failed for '{keyword}': {e}")
        return None

def mock_keyword_score(keyword: str) -> Dict[str, Any]:
    """Fallback mock scoring when SerpAPI is not available."""
    word_count = len(keyword.split())
    
    # Simulate realistic patterns
    if "how to" in keyword.lower() or "?" in keyword:
        competition_base = 0.2
        volume_base = 800
    elif word_count >= 4:
        competition_base = 0.15
        volume_base = 300
    elif word_count == 3:
        competition_base = 0.35
        volume_base = 1500
    elif word_count == 2:
        competition_base = 0.5
        volume_base = 5000
    else:
        competition_base = 0.7
        volume_base = 10000
    
    # Add some variation
    import random
    competition_score = competition_base + random.uniform(-0.1, 0.1)
    competition_score = max(0.1, min(0.9, competition_score))
    estimated_volume = int(volume_base * random.uniform(0.5, 1.5))
    
    volume_score = math.pow(estimated_volume, 0.7)
    opportunity_score = volume_score / (competition_score + 0.1)
    
    return {
        "keyword": keyword,
        "monthly_searches": estimated_volume,
        "competition_score": round(competition_score, 4),
        "opportunity_score": round(opportunity_score, 2),
        "difficulty": "Easy" if competition_score < 0.3 else "Medium" if competition_score < 0.6 else "Hard",
        "first_page_potential": competition_score < 0.5,
        "ranking_chance": "High" if competition_score < 0.3 else "Medium" if competition_score < 0.5 else "Low",
        "total_results": int(estimated_volume * 1000 * competition_score),
        "ads_count": min(int(competition_score * 5), 4),
        "high_da_competitors": int(competition_score * 10),
        "easy_targets": max(0, 10 - int(competition_score * 10)),
        "featured_snippet": "No",
        "people_also_ask": "Yes" if word_count > 2 else "No",
        "knowledge_graph": "No",
        "local_pack": "No"
    }

def score_candidates_for_first_page(candidates: List[str], target_count: int = 50) -> List[Dict[str, Any]]:
    """
    Score candidates in parallel, focusing on finding the best 50 for first page ranking.
    """
    if not candidates:
        return []
    
    scored_results = []
    failed_keywords = []
    
    # Process in batches for efficiency
    batch_size = 5
    total_batches = math.ceil(len(candidates) / batch_size)
    
    logger.info(f"Processing {len(candidates)} candidates in {total_batches} batches")
    
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        try:
            # Process batch with timeout
            futures = [thread_pool.submit(score_keyword_for_first_page, keyword) for keyword in batch]
            
            for future, keyword in zip(futures, batch):
                try:
                    result = future.result(timeout=15)
                    if result:
                        scored_results.append(result)
                    else:
                        failed_keywords.append(keyword)
                except Exception as e:
                    logger.warning(f"Timeout/error for '{keyword}': {e}")
                    failed_keywords.append(keyword)
            
            # Small delay between batches to avoid rate limiting
            if batch_num < total_batches:
                time.sleep(0.5)
            
            # Log progress
            if batch_num % 5 == 0:
                logger.info(f"Processed {batch_num}/{total_batches} batches, found {len(scored_results)} valid results")
            
        except Exception as e:
            logger.error(f"Batch {batch_num} processing failed: {e}")
            continue
    
    if failed_keywords:
        logger.warning(f"Failed to score {len(failed_keywords)} keywords")
    
    # Sort by best opportunity for first page ranking
    # Primary: First page potential, Secondary: Low competition, Tertiary: High opportunity score
    scored_results.sort(
        key=lambda x: (
            not x.get("first_page_potential", False),  # First page potential first
            x.get("competition_score", 1.0),  # Then lowest competition
            -x.get("opportunity_score", 0),  # Then highest opportunity
        )
    )
    
    # Get top results
    top_results = scored_results[:target_count]
    
    # Log summary
    first_page_count = sum(1 for r in top_results if r.get("first_page_potential", False))
    avg_competition = sum(r.get("competition_score", 0) for r in top_results) / len(top_results) if top_results else 0
    
    logger.info(f"Found {len(top_results)} keywords: {first_page_count} with first page potential, "
                f"avg competition: {avg_competition:.3f}")
    
    return top_results

def check_rate_limit(client_ip: str) -> bool:
    """Rate limiting check."""
    current_time = time.time()
    
    if client_ip not in REQUEST_TIMES:
        REQUEST_TIMES[client_ip] = []
    
    REQUEST_TIMES[client_ip] = [
        req_time for req_time in REQUEST_TIMES[client_ip]
        if current_time - req_time < RATE_LIMIT_WINDOW
    ]
    
    if len(REQUEST_TIMES[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return False
    
    REQUEST_TIMES[client_ip].append(current_time)
    return True

def get_cache_key(seed: str, top: int, focus: str) -> str:
    """Generate cache key."""
    return f"{seed.lower().strip()}_{top}_{focus}"

def check_cache(cache_key: str) -> Optional[Dict]:
    """Check cache."""
    if cache_key in REQUEST_CACHE:
        cached_data, timestamp = REQUEST_CACHE[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return cached_data
        else:
            del REQUEST_CACHE[cache_key]
    return None

def set_cache(cache_key: str, data: Dict):
    """Set cache."""
    REQUEST_CACHE[cache_key] = (data, time.time())
    
    # Limit cache size
    if len(REQUEST_CACHE) > 100:
        # Remove oldest entries
        sorted_cache = sorted(REQUEST_CACHE.items(), key=lambda x: x[1][1])
        for key, _ in sorted_cache[:20]:
            del REQUEST_CACHE[key]

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        processing_time = time.time() - start_time
        response.headers["X-Processing-Time"] = str(round(processing_time, 3))
        
        return response
        
    except Exception as e:
        logger.error(f"Middleware error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("SEO First Page Keyword API starting up...")
    logger.info(f"SerpAPI available: {HAS_SERPAPI and bool(SERPAPI_KEY)}")
    logger.info(f"Target: 50 low-competition keywords per request")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("API shutting down...")
    thread_pool.shutdown(wait=False)

# Main endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SEO Keyword Research API - First Page Optimizer",
        "version": "3.0.0",
        "description": "Find 50 low-competition, high-volume keywords with first page ranking potential",
        "endpoints": {
            "/keywords": "Main keyword research endpoint",
            "/health": "Health check",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "serpapi_enabled": HAS_SERPAPI and bool(SERPAPI_KEY),
        "cache_size": len(REQUEST_CACHE)
    }

@app.get("/keywords", response_model=KeywordResponse)
async def get_keywords(
    request: Request,
    seed: str = Query(..., description="Seed keyword for research"),
    top: int = Query(50, description="Number of results (default: 50)", ge=10, le=100),
    focus: str = Query("first_page", description="Focus: first_page, volume, or balanced")
):
    """
    Main keyword research endpoint.
    Returns up to 50 keywords sorted by lowest competition and highest search volume.
    Focuses on keywords with realistic first page ranking potential.
    """
    # Authentication check
    if API_AUTH_KEY:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(401, "API key required")
        if auth_header.replace("Bearer ", "") != API_AUTH_KEY:
            raise HTTPException(401, "Invalid API key")
    
    # Rate limiting
    client_ip = request.client.host or "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(429, "Rate limit exceeded. Please wait before making another request.")
    
    start_time = time.time()
    
    # Validate input
    seed = seed.strip()
    if not seed or len(seed) > 100:
        raise HTTPException(400, "Invalid seed keyword. Must be 1-100 characters.")
    
    # Check cache
    cache_key = get_cache_key(seed, top, focus)
    cached_result = check_cache(cache_key)
    if cached_result:
        logger.info(f"Cache hit for: '{seed}'")
        return KeywordResponse(**cached_result, cached=True)
    
    try:
        logger.info(f"Processing request: seed='{seed}', top={top}, focus={focus}")
        
        # Step 1: Collect comprehensive candidates (2x the target for better selection)
        max_candidates = min(top * 2, 200)
        candidates = collect_candidates_comprehensive(seed, max_candidates)
        
        if not candidates:
            raise HTTPException(404, "No keyword candidates found. Try a different seed keyword.")
        
        logger.info(f"Collected {len(candidates)} candidates")
        
        # Step 2: Score all candidates for first page potential
        scored_results = score_candidates_for_first_page(candidates, target_count=top)
        
        if not scored_results:
            raise HTTPException(500, "Failed to score keywords. Please try again.")
        
        # Step 3: Apply focus-based sorting if needed
        if focus == "volume":
            # Re-sort by volume while maintaining some competition awareness
            scored_results.sort(
                key=lambda x: (
                    -x.get("monthly_searches", 0),  # Highest volume first
                    x.get("competition_score", 1.0)  # Then lowest competition
                )
            )
        elif focus == "balanced":
            # Already sorted by opportunity score (balanced approach)
            scored_results.sort(
                key=lambda x: -x.get("opportunity_score", 0)
            )
        # else: keep "first_page" focus (default sorting)
        
        # Get final results
        final_results = scored_results[:top]
        
        # Calculate statistics
        first_page_count = sum(1 for r in final_results if r.get("first_page_potential", False))
        avg_competition = sum(r.get("competition_score", 0) for r in final_results) / len(final_results) if final_results else 0
        avg_volume = sum(r.get("monthly_searches", 0) for r in final_results) / len(final_results) if final_results else 0
        
        processing_time = time.time() - start_time
        
        # Prepare response
        result_data = {
            "seed": seed,
            "returned": len(final_results),
            "results": final_results,
            "processing_time_seconds": round(processing_time, 2),
            "first_page_potential": first_page_count
        }
        
        # Cache the results
        set_cache(cache_key, result_data)
        
        # Log summary
        logger.info(f"Completed: {len(final_results)} keywords, {first_page_count} with first page potential, "
                   f"avg competition: {avg_competition:.3f}, avg volume: {avg_volume:.0f}, "
                   f"time: {processing_time:.2f}s")
        
        return KeywordResponse(**result_data, cached=False)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.post("/analyze")
async def analyze_keywords(
    request: Request,
    seed: str = Query(..., description="Seed keyword"),
    background_tasks: BackgroundTasks = None
):
    """
    Analyze keywords with detailed competition metrics.
    Returns comprehensive analysis for first page ranking strategy.
    """
    # Authentication
    if API_AUTH_KEY:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(401, "API key required")
        if auth_header.replace("Bearer ", "") != API_AUTH_KEY:
            raise HTTPException(401, "Invalid API key")
    
    seed = seed.strip()
    if not seed:
        raise HTTPException(400, "Seed keyword is required")
    
    try:
        # Quick analysis of the seed keyword itself
        logger.info(f"Analyzing seed keyword: '{seed}'")
        
        # Get detailed analysis
        seed_analysis = score_keyword_for_first_page(seed)
        
        if not seed_analysis:
            raise HTTPException(500, "Analysis failed")
        
        # Get related keywords for comparison
        candidates = collect_candidates_comprehensive(seed, max_candidates=20)
        related_analysis = score_candidates_for_first_page(candidates[:10], target_count=10)
        
        # Prepare analysis summary
        analysis_result = {
            "seed_keyword": seed_analysis,
            "recommendations": [],
            "related_keywords": related_analysis,
            "strategy": ""
        }
        
        # Generate recommendations based on competition
        competition = seed_analysis.get("competition_score", 1.0)
        
        if competition < 0.3:
            analysis_result["strategy"] = "AGGRESSIVE: Low competition detected. Target this keyword directly with comprehensive content."
            analysis_result["recommendations"] = [
                "Create a comprehensive, 2000+ word guide",
                "Include FAQ section to capture featured snippets",
                "Add schema markup for better SERP visibility",
                "Build 5-10 high-quality backlinks",
                "Optimize for the exact keyword in title and H1"
            ]
        elif competition < 0.5:
            analysis_result["strategy"] = "MODERATE: Medium competition. Focus on long-tail variations and content depth."
            analysis_result["recommendations"] = [
                "Target long-tail variations primarily",
                "Create multiple pieces of supporting content",
                "Focus on user intent and comprehensive answers",
                "Build topical authority with related content",
                "Acquire 10-20 relevant backlinks over time"
            ]
        elif competition < 0.7:
            analysis_result["strategy"] = "CONSERVATIVE: High competition. Build authority first with easier keywords."
            analysis_result["recommendations"] = [
                "Start with long-tail, question-based keywords",
                "Build topical authority gradually",
                "Create a content hub around the topic",
                "Focus on local or niche variations",
                "Establish expertise before targeting main keyword"
            ]
        else:
            analysis_result["strategy"] = "ALTERNATIVE: Very high competition. Consider different approach or keywords."
            analysis_result["recommendations"] = [
                "Target completely different long-tail variations",
                "Focus on local SEO if applicable",
                "Consider paid advertising instead",
                "Build brand searches as alternative strategy",
                "Create unique angle or perspective on topic"
            ]
        
        return {
            "status": "completed",
            "analysis": analysis_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@app.get("/export")
async def export_results(
    request: Request,
    seed: str = Query(..., description="Seed keyword used in previous search"),
    format: str = Query("json", description="Export format: json or csv")
):
    """
    Export keyword research results in different formats.
    """
    # Check cache for recent results
    cache_key = get_cache_key(seed, 50, "first_page")
    cached_result = check_cache(cache_key)
    
    if not cached_result:
        raise HTTPException(404, "No recent results found for this seed keyword. Please run a search first.")
    
    results = cached_result.get("results", [])
    
    if format == "csv":
        if not HAS_PANDAS:
            raise HTTPException(500, "CSV export not available (pandas not installed)")
        
        # Convert to CSV
        df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        column_order = [
            "keyword", "monthly_searches", "competition_score", "opportunity_score",
            "difficulty", "first_page_potential", "ranking_chance", "total_results",
            "ads_count", "high_da_competitors", "easy_targets", "featured_snippet",
            "people_also_ask", "knowledge_graph", "local_pack"
        ]
        
        df = df[[col for col in column_order if col in df.columns]]
        
        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=keywords_{seed.replace(' ', '_')}.csv"}
        )
    
    else:
        # Return as JSON
        return {
            "seed": seed,
            "exported_at": datetime.utcnow().isoformat(),
            "count": len(results),
            "results": results
        }

@app.get("/stats")
async def get_statistics():
    """
    Get API usage statistics and performance metrics.
    """
    total_cached = len(REQUEST_CACHE)
    
    # Calculate cache hit rate
    cache_stats = {
        "total_cached_queries": total_cached,
        "cache_memory_usage": sum(len(str(v)) for v in REQUEST_CACHE.values()) if REQUEST_CACHE else 0,
        "rate_limited_ips": len(REQUEST_TIMES)
    }
    
    return {
        "status": "operational",
        "statistics": cache_stats,
        "configuration": {
            "default_target_keywords": 50,
            "max_candidates_analyzed": 200,
            "rate_limit": f"{RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds",
            "cache_ttl": f"{CACHE_TTL} seconds"
        },
        "thresholds": FIRST_PAGE_THRESHOLDS,
        "timestamp": datetime.utcnow().isoformat()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting SEO First Page Keyword API on port {port}")
    logger.info("Target: Find 50 keywords with lowest competition and highest search volume")
    logger.info("Focus: Keywords with realistic first page ranking potential")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,  # Single worker for stability
        log_level="info",
        access_log=True
    )