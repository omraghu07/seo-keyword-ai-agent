# src/server.py
"""
Free-Plan Friendly SEO Keyword Research API
Optimized to minimize SerpAPI calls while maximizing keyword discovery

Key Features:
- Configurable keyword count (5, 10, 20, 50, etc.)
- Only 1 SerpAPI call per seed for candidate collection
- Mock scoring for initial ranking
- Optional SerpAPI verification for top N results
- Strict mode for free plan protection (max 5 API calls per request)
"""

import os
import logging
import time
import math
import re
import io
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import Counter

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
    try:
        from google_search_results import GoogleSearch
        HAS_SERPAPI = True
    except ImportError:
        HAS_SERPAPI = False

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Free-Plan Friendly SEO Keyword API",
    description="Efficient keyword research optimized for SerpAPI free plan",
    version="4.0.0",
    docs_url="/docs"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
API_AUTH_KEY = os.getenv("API_AUTH_KEY")
USE_SERPAPI_STRICT_MODE = os.getenv("USE_SERPAPI_STRICT_MODE", "true").lower() == "true"
MAX_SERPAPI_CALLS_STRICT = 5  # Maximum API calls in strict mode
MAX_SERPAPI_CALLS_NORMAL = 20  # Maximum API calls in normal mode

# Rate limiting
REQUEST_TIMES = {}
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX_REQUESTS = 30

# Request counter for monitoring
API_CALL_COUNTER = {"total": 0, "session_start": time.time()}

class KeywordResponse(BaseModel):
    """API response model."""
    success: bool = True
    seed: str
    requested: int
    returned: int
    results: List[Dict[str, Any]]
    processing_time: float
    api_calls_used: int
    api_budget_remaining: int
    data_source: str
    timestamp: str

def count_api_call():
    """Track API usage."""
    API_CALL_COUNTER["total"] += 1
    logger.info(f"API call #{API_CALL_COUNTER['total']} - Session time: {time.time() - API_CALL_COUNTER['session_start']:.1f}s")

def get_api_budget() -> int:
    """Calculate remaining API budget for this request."""
    max_calls = MAX_SERPAPI_CALLS_STRICT if USE_SERPAPI_STRICT_MODE else MAX_SERPAPI_CALLS_NORMAL
    used = API_CALL_COUNTER["total"]
    return max(0, max_calls - used)

def heuristic_competition_score(keyword: str) -> float:
    """
    Calculate mock competition score based on keyword characteristics.
    Does NOT use any API calls.
    """
    words = keyword.lower().split()
    word_count = len(words)
    
    # Base competition by word count
    base_scores = {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.25, 5: 0.2}
    base_score = base_scores.get(word_count, max(0.15, 0.3 - (word_count * 0.02)))
    
    # Adjust for question keywords (lower competition)
    question_words = ["how", "what", "why", "when", "where", "who", "which", "can", "should", "is", "are", "does"]
    if any(word in words for word in question_words):
        base_score *= 0.7
    
    # Adjust for commercial intent (higher competition)
    commercial_words = ["buy", "best", "top", "review", "price", "cheap", "discount"]
    if any(word in words for word in commercial_words):
        base_score *= 1.3
    
    # Adjust for specific/niche keywords (lower competition)
    specific_words = ["beginner", "tutorial", "guide", "explained", "step", "diy", "simple"]
    if any(word in words for word in specific_words):
        base_score *= 0.8
    
    # Add some deterministic variation based on keyword hash
    variation = (hash(keyword) % 20) / 100  # -0.1 to +0.1
    base_score += variation
    
    return max(0.05, min(0.95, base_score))

def heuristic_search_volume(keyword: str) -> int:
    """
    Estimate search volume based on keyword characteristics.
    Does NOT use any API calls.
    """
    words = keyword.lower().split()
    word_count = len(words)
    
    # Base volumes
    base_volumes = {1: 10000, 2: 5000, 3: 2000, 4: 800, 5: 400}
    base_volume = base_volumes.get(word_count, max(100, 500 - (word_count * 50)))
    
    # Adjust for popular terms
    popular_terms = ["free", "online", "best", "how", "tutorial", "guide"]
    if any(term in words for term in popular_terms):
        base_volume = int(base_volume * 1.5)
    
    # Adjust for very specific/niche terms
    niche_terms = ["advanced", "professional", "enterprise", "custom"]
    if any(term in words for term in niche_terms):
        base_volume = int(base_volume * 0.6)
    
    # Add deterministic variation
    variation_factor = 1 + ((hash(keyword) % 40) - 20) / 100  # 0.8 to 1.2
    volume = int(base_volume * variation_factor)
    
    return max(10, min(100000, volume))

def calculate_opportunity_score(volume: int, competition: float) -> float:
    """Calculate opportunity score."""
    volume_score = math.log10(volume + 1)
    return volume_score / (competition + 0.1)

def score_keyword_heuristic(keyword: str) -> Dict[str, Any]:
    """
    Score a keyword using only heuristics (NO API calls).
    Fast and free method for initial ranking.
    """
    competition = heuristic_competition_score(keyword)
    volume = heuristic_search_volume(keyword)
    opportunity = calculate_opportunity_score(volume, competition)
    
    # Determine difficulty
    if competition < 0.3:
        difficulty = "Easy"
    elif competition < 0.5:
        difficulty = "Medium"
    elif competition < 0.7:
        difficulty = "Hard"
    else:
        difficulty = "Very Hard"
    
    # Estimate ranking potential
    if competition < 0.4 and volume >= 300:
        ranking_chance = "High"
    elif competition < 0.6 and volume >= 100:
        ranking_chance = "Medium"
    else:
        ranking_chance = "Low"
    
    return {
        "keyword": keyword,
        "monthly_searches": volume,
        "competition_score": round(competition, 4),
        "opportunity_score": round(opportunity, 2),
        "difficulty": difficulty,
        "ranking_chance": ranking_chance,
        "data_source": "heuristic"
    }

def enrich_with_serpapi(keyword: str) -> Optional[Dict[str, Any]]:
    """
    Enrich a keyword with real SerpAPI data.
    Uses 1 API call per keyword.
    """
    if not HAS_SERPAPI or not SERPAPI_KEY:
        logger.warning("SerpAPI not available for enrichment")
        return None
    
    try:
        count_api_call()
        
        params = {
            "engine": "google",
            "q": keyword,
            "api_key": SERPAPI_KEY,
            "hl": "en",
            "gl": "us",
            "num": 10
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "error" in results:
            logger.error(f"SerpAPI error: {results['error']}")
            return None
        
        # Extract metrics
        search_info = results.get("search_information", {})
        total_results_raw = search_info.get("total_results") or search_info.get("total_results_raw") or ""
        total_results = 0
        if isinstance(total_results_raw, int):
            total_results = total_results_raw
        elif isinstance(total_results_raw, str):
            nums = re.sub(r"[^\d]", "", total_results_raw)
            total_results = int(nums) if nums else 0
        
        ads_count = len(results.get("ads_results", []))
        has_featured_snippet = bool(results.get("featured_snippet") or results.get("answer_box"))
        has_paa = bool(results.get("related_questions") or results.get("people_also_ask"))
        has_kg = bool(results.get("knowledge_graph"))
        
        # Calculate real competition
        normalized_results = min(math.log10(total_results + 1) / 7, 1.0) if total_results > 0 else 0
        ads_score = min(ads_count / 3, 1.0)
        
        competition = (
            0.40 * normalized_results +
            0.25 * ads_score +
            0.15 * (1 if has_featured_snippet else 0) +
            0.10 * (1 if has_paa else 0) +
            0.10 * (1 if has_kg else 0)
        )
        competition = max(0.0, min(1.0, competition))
        
        # Estimate volume from signals
        word_count = len(keyword.split())
        base_volume = max(100, 8000 // (word_count + 1))
        
        if ads_count > 2:
            base_volume = int(base_volume * 1.5)
        if has_featured_snippet:
            base_volume = int(base_volume * 1.2)
        
        volume = min(base_volume, 50000)
        opportunity = calculate_opportunity_score(volume, competition)
        
        # Determine difficulty
        if competition < 0.3:
            difficulty = "Easy"
        elif competition < 0.5:
            difficulty = "Medium"
        elif competition < 0.7:
            difficulty = "Hard"
        else:
            difficulty = "Very Hard"
        
        # Ranking chance
        if competition < 0.35:
            ranking_chance = "High"
        elif competition < 0.55:
            ranking_chance = "Medium"
        else:
            ranking_chance = "Low"
        
        return {
            "keyword": keyword,
            "monthly_searches": volume,
            "competition_score": round(competition, 4),
            "opportunity_score": round(opportunity, 2),
            "difficulty": difficulty,
            "ranking_chance": ranking_chance,
            "total_results": total_results,
            "ads_count": ads_count,
            "featured_snippet": "Yes" if has_featured_snippet else "No",
            "people_also_ask": "Yes" if has_paa else "No",
            "knowledge_graph": "Yes" if has_kg else "No",
            "data_source": "serpapi"
        }
        
    except Exception as e:
        logger.error(f"SerpAPI enrichment failed for '{keyword}': {e}")
        return None

def collect_candidates_from_seed(seed: str) -> Tuple[List[str], int]:
    """
    Collect keyword candidates using ONLY 1 SerpAPI call.
    Returns (candidates, api_calls_used)
    """
    candidates = set()
    candidates.add(seed)  # Always include seed
    api_calls = 0
    
    # Generate synthetic candidates (NO API calls)
    question_words = ["how to", "what is", "why", "when", "where", "can i", "should i"]
    modifiers = ["best", "free", "online", "guide", "tutorial", "tips", "examples", 
                 "for beginners", "explained", "2024", "2025", "cheap", "review"]
    
    for q in question_words[:5]:
        candidates.add(f"{q} {seed}")
    
    for mod in modifiers[:15]:
        candidates.add(f"{seed} {mod}")
        candidates.add(f"{mod} {seed}")
    
    # Make ONE SerpAPI call to get real related keywords
    if HAS_SERPAPI and SERPAPI_KEY:
        try:
            count_api_call()
            api_calls = 1
            
            params = {
                "engine": "google",
                "q": seed,
                "api_key": SERPAPI_KEY,
                "hl": "en",
                "gl": "us"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "error" not in results:
                # Extract related searches
                for item in results.get("related_searches", [])[:20]:
                    query = item.get("query", "")
                    if query and len(query.split()) <= 6:
                        candidates.add(query.lower().strip())
                
                # Extract PAA questions
                for item in results.get("related_questions", [])[:15]:
                    question = item.get("question", "")
                    if question:
                        candidates.add(question.lower().strip())
                
                logger.info(f"SerpAPI call successful: collected real suggestions")
            else:
                logger.warning(f"SerpAPI error: {results.get('error')}")
                
        except Exception as e:
            logger.error(f"SerpAPI collection failed: {e}")
    
    final_candidates = list(candidates)
    logger.info(f"Collected {len(final_candidates)} candidates ({api_calls} API call)")
    
    return final_candidates, api_calls

def check_rate_limit(client_ip: str) -> bool:
    """Rate limiting."""
    current_time = time.time()
    
    if client_ip not in REQUEST_TIMES:
        REQUEST_TIMES[client_ip] = []
    
    REQUEST_TIMES[client_ip] = [
        t for t in REQUEST_TIMES[client_ip]
        if current_time - t < RATE_LIMIT_WINDOW
    ]
    
    if len(REQUEST_TIMES[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return False
    
    REQUEST_TIMES[client_ip].append(current_time)
    return True

@app.on_event("startup")
async def startup():
    """Startup logging."""
    logger.info("=" * 60)
    logger.info("SEO Keyword API - Free Plan Optimized")
    logger.info(f"Strict Mode: {USE_SERPAPI_STRICT_MODE}")
    logger.info(f"Max API calls per request: {MAX_SERPAPI_CALLS_STRICT if USE_SERPAPI_STRICT_MODE else MAX_SERPAPI_CALLS_NORMAL}")
    logger.info(f"SerpAPI Available: {HAS_SERPAPI and bool(SERPAPI_KEY)}")
    logger.info("=" * 60)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Free-Plan Friendly SEO Keyword API",
        "version": "4.0.0",
        "strict_mode": USE_SERPAPI_STRICT_MODE,
        "max_api_calls": MAX_SERPAPI_CALLS_STRICT if USE_SERPAPI_STRICT_MODE else MAX_SERPAPI_CALLS_NORMAL,
        "strategy": "1 API call for candidate collection + optional enrichment for top N",
        "endpoints": {
            "/keywords": "Main keyword research (configurable count)",
            "/health": "Health check",
            "/stats": "API usage statistics"
        }
    }

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "serpapi_available": HAS_SERPAPI and bool(SERPAPI_KEY),
        "strict_mode": USE_SERPAPI_STRICT_MODE,
        "session_api_calls": API_CALL_COUNTER["total"]
    }

@app.get("/stats")
async def stats():
    """API usage statistics."""
    uptime = time.time() - API_CALL_COUNTER["session_start"]
    return {
        "session_start": datetime.fromtimestamp(API_CALL_COUNTER["session_start"]).isoformat(),
        "uptime_seconds": round(uptime, 1),
        "total_api_calls": API_CALL_COUNTER["total"],
        "strict_mode": USE_SERPAPI_STRICT_MODE,
        "max_calls_per_request": MAX_SERPAPI_CALLS_STRICT if USE_SERPAPI_STRICT_MODE else MAX_SERPAPI_CALLS_NORMAL
    }

@app.get("/keywords", response_model=KeywordResponse)
async def get_keywords(
    request: Request,
    seed: str = Query(..., description="Seed keyword", min_length=1, max_length=100),
    top: int = Query(30, description="Number of keywords to return", ge=1, le=100),
    enrich_top: int = Query(10, description="Number of top results to enrich with SerpAPI", ge=0, le=20)
):
    """
    Main keyword research endpoint.
    
    Strategy:
    1. Make 1 SerpAPI call to collect candidates from seed
    2. Score all candidates with heuristics (free)
    3. Optionally enrich top N with real SerpAPI data
    
    Parameters:
    - seed: Your main keyword
    - top: How many keywords you want (e.g., 5, 10, 20, 50)
    - enrich_top: How many of the top results to verify with SerpAPI (0 = none, saves API calls)
    
    Example: top=10, enrich_top=3 means:
    - 1 API call to collect candidates
    - Return 10 keywords scored with heuristics
    - Enrich the top 3 with real SerpAPI data (3 more API calls)
    - Total: 4 API calls
    """
    start_time = time.time()
    client_ip = request.client.host or "unknown"
    
    # Authentication
    if API_AUTH_KEY:
        auth = request.headers.get("Authorization", "").replace("Bearer ", "")
        if auth != API_AUTH_KEY:
            raise HTTPException(401, "Invalid or missing API key")
    
    # Rate limiting
    if not check_rate_limit(client_ip):
        raise HTTPException(429, "Rate limit exceeded")
    
    # Validate
    seed = seed.strip().lower()
    if not seed:
        raise HTTPException(400, "Invalid seed keyword")
    
    # Check API budget
    max_calls = MAX_SERPAPI_CALLS_STRICT if USE_SERPAPI_STRICT_MODE else MAX_SERPAPI_CALLS_NORMAL
    if enrich_top > 0:
        required_calls = 1 + enrich_top  # 1 for collection + N for enrichment
        if required_calls > max_calls:
            raise HTTPException(
                400,
                f"Request would use {required_calls} API calls, but budget is {max_calls}. "
                f"Reduce enrich_top to {max_calls - 1} or less."
            )
    
    try:
        logger.info(f"Request: seed='{seed}', top={top}, enrich_top={enrich_top}")
        
        # Step 1: Collect candidates (1 API call)
        candidates, api_calls_used = collect_candidates_from_seed(seed)
        
        if not candidates:
            raise HTTPException(404, "No candidates found")
        
        # Step 2: Score all candidates with heuristics (FREE - no API calls)
        logger.info(f"Scoring {len(candidates)} candidates with heuristics...")
        scored_candidates = []
        for candidate in candidates:
            try:
                result = score_keyword_heuristic(candidate)
                scored_candidates.append(result)
            except Exception as e:
                logger.warning(f"Heuristic scoring failed for '{candidate}': {e}")
                continue
        
        # Sort by opportunity score (highest first)
        scored_candidates.sort(key=lambda x: x["opportunity_score"], reverse=True)
        
        # Get top N requested
        top_results = scored_candidates[:top]
        
        # Step 3: Optionally enrich top results with real SerpAPI data
        data_source = "heuristic"
        if enrich_top > 0 and HAS_SERPAPI and SERPAPI_KEY:
            logger.info(f"Enriching top {enrich_top} results with SerpAPI...")
            
            for i in range(min(enrich_top, len(top_results))):
                keyword = top_results[i]["keyword"]
                
                # Check budget before each call
                if api_calls_used >= max_calls:
                    logger.warning(f"API budget exhausted at {api_calls_used} calls")
                    break
                
                enriched = enrich_with_serpapi(keyword)
                if enriched:
                    top_results[i] = enriched
                    api_calls_used += 1
                    data_source = "mixed"
                    
                # Small delay between calls
                time.sleep(0.2)
            
            logger.info(f"Enrichment complete: {api_calls_used} total API calls used")
        
        # Add ranking
        for rank, result in enumerate(top_results, 1):
            result["rank"] = rank
        
        processing_time = time.time() - start_time
        budget_remaining = max_calls - api_calls_used
        
        logger.info(
            f"SUCCESS: Returned {len(top_results)} keywords, "
            f"API calls: {api_calls_used}/{max_calls}, "
            f"Time: {processing_time:.2f}s"
        )
        
        return KeywordResponse(
            success=True,
            seed=seed,
            requested=top,
            returned=len(top_results),
            results=top_results,
            processing_time=round(processing_time, 2),
            api_calls_used=api_calls_used,
            api_budget_remaining=budget_remaining,
            data_source=data_source,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.get("/export/csv")
async def export_csv(
    seed: str = Query(...),
    top: int = Query(50),
    enrich_top: int = Query(0)
):
    """Export results as CSV."""
    if not HAS_PANDAS:
        raise HTTPException(500, "CSV export unavailable (pandas not installed)")
    
    # Get keyword data
    response = await get_keywords(Request(scope={"type": "http", "client": ("127.0.0.1", 0), "headers": []}), seed, top, enrich_top)
    
    # Convert to DataFrame
    df = pd.DataFrame(response.results)
    
    # Create CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=keywords_{seed.replace(' ', '_')}.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )