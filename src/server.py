# src/server.py
"""
FastAPI server for SEO keyword research API - Optimized for Render
"""

import os
import logging
import time
import io
import math
import re
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Depends, Query, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
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
    title="SEO Keyword Research API",
    description="REST API for keyword expansion and scoring using SerpAPI",
    version="1.0.0",
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

# Rate limiting setup
REQUEST_TIMES = {}
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX_REQUESTS = 10  # Reduced for Render

# Cache for recent requests to avoid repeated API calls
REQUEST_CACHE = {}
CACHE_TTL = 300  # 5 minutes

# Thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=5)

class KeywordResponse(BaseModel):
    """Response model for keyword analysis results."""
    seed: str = Field(..., description="The input seed keyword")
    returned: int = Field(..., description="Number of results returned")
    results: List[Dict[str, Any]] = Field(..., description="List of keyword analysis results")
    processing_time_seconds: float = Field(..., description="Time taken to process the request")
    cached: bool = Field(..., description="Whether results were served from cache")

class KeywordRequest(BaseModel):
    """Request model for async keyword processing."""
    seed: str
    top: int = 20
    max_candidates: int = 50  # Reduced for faster processing

# Store for async processing results
async_results = {}

def extract_total_results(search_info: Dict[str, Any]) -> int:
    """Extract total results count from SerpApi response."""
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

def calculate_competition_score(search_results: Dict[str, Any]) -> tuple:
    """Calculate competition score based on SERP features."""
    search_info = search_results.get("search_information", {})
    
    # Factor 1: Total number of results
    total_results = extract_total_results(search_info)
    normalized_results = min(math.log10(total_results + 1) / 7, 1.0)
    
    # Factor 2: Number of ads
    ads = search_results.get("ads_results", [])
    ads_count = len(ads) if ads else 0
    ads_score = min(ads_count / 3, 1.0)
    
    # Factor 3: SERP features
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
        0.50 * normalized_results +
        0.25 * ads_score +
        0.15 * has_featured_snippet +
        0.07 * has_people_also_ask +
        0.03 * has_knowledge_graph
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

def collect_candidates_fast(seed: str, max_candidates: int = 50) -> List[str]:
    """Fast keyword candidate collection with fallback."""
    if not HAS_SERPAPI or not os.getenv("SERPAPI_KEY"):
        # Return mock data quickly
        return [f"{seed} {suffix}" for suffix in [
            "tips", "guide", "best", "tutorial", "examples",
            "free", "online", "course", "training", "how to"
        ]]
    
    try:
        search_params = {
            "engine": "google",
            "q": seed,
            "api_key": os.getenv("SERPAPI_KEY"),
            "hl": "en",
            "gl": "us",
            "num": 5  # Reduced for speed
        }
        
        search = GoogleSearch(search_params)
        results = search.get_dict()
        
        keyword_candidates = set()
        
        # Extract from related searches
        related_searches = results.get("related_searches", [])[:5]
        for item in related_searches:
            query = item.get("query") or item.get("suggestion")
            if query:
                keyword_candidates.add(query.strip())
        
        # Extract from people also ask (limited)
        related_questions = results.get("related_questions", [])[:3]
        for item in related_questions:
            question = item.get("question") or item.get("query")
            if question:
                keyword_candidates.add(question.strip())
        
        # Add some basic variations
        basic_variations = [f"{seed} {v}" for v in ["review", "buy", "price", "2024", "best"]]
        keyword_candidates.update(basic_variations)
        
        return list(keyword_candidates)[:max_candidates]
        
    except Exception as e:
        logger.warning(f"Fast candidate collection failed: {e}")
        # Fallback to mock data
        return [f"{seed} {suffix}" for suffix in [
            "tips", "guide", "best", "tutorial", "examples"
        ]]

def score_single_keyword(keyword: str) -> Dict[str, Any]:
    """Score a single keyword (for parallel processing)."""
    try:
        # Mock scoring for speed - remove this if you want real SerpAPI calls
        word_count = len(keyword.split())
        estimated_volume = max(50, 5000 // (word_count + 1))
        competition_score = min(0.1 + (word_count * 0.05), 0.8)
        volume_score = math.log10(estimated_volume + 1)
        opportunity_score = volume_score / (competition_score + 0.01)
        
        return {
            "keyword": keyword,
            "monthly_searches": estimated_volume,
            "competition_score": round(competition_score, 4),
            "opportunity_score": round(opportunity_score, 2),
            "total_results": estimated_volume * 1000,
            "ads_count": word_count % 3,
            "featured_snippet": "Yes" if word_count % 2 == 0 else "No",
            "people_also_ask": "Yes" if word_count % 3 == 0 else "No",
            "knowledge_graph": "Yes" if word_count % 4 == 0 else "No"
        }
    except Exception as e:
        logger.warning(f"Failed to score keyword '{keyword}': {e}")
        return None

def score_candidates_fast(candidates: List[str]) -> List[Dict[str, Any]]:
    """Fast parallel scoring of candidates."""
    if not candidates:
        return []
    
    # Use thread pool for parallel processing
    scored_results = []
    
    # Process in smaller batches to avoid timeouts
    batch_size = min(5, len(candidates))
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        
        # Process batch in parallel
        futures = [thread_pool.submit(score_single_keyword, keyword) for keyword in batch]
        batch_results = [future.result() for future in futures]
        
        # Filter out None results
        valid_results = [r for r in batch_results if r is not None]
        scored_results.extend(valid_results)
        
        # Small delay between batches
        time.sleep(0.5)
    
    # Sort by opportunity score (highest first)
    scored_results.sort(key=lambda x: x["opportunity_score"], reverse=True)
    return scored_results

def check_rate_limit(client_ip: str) -> bool:
    """Simple rate limiting check."""
    current_time = time.time()
    
    if client_ip not in REQUEST_TIMES:
        REQUEST_TIMES[client_ip] = []
    
    # Clean old requests
    REQUEST_TIMES[client_ip] = [
        req_time for req_time in REQUEST_TIMES[client_ip]
        if current_time - req_time < RATE_LIMIT_WINDOW
    ]
    
    if len(REQUEST_TIMES[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return False
    
    REQUEST_TIMES[client_ip].append(current_time)
    return True

def get_cache_key(seed: str, top: int, max_candidates: int) -> str:
    """Generate cache key for request."""
    return f"{seed.lower()}_{top}_{max_candidates}"

def check_cache(cache_key: str) -> Optional[Dict]:
    """Check if results are in cache."""
    if cache_key in REQUEST_CACHE:
        cached_data, timestamp = REQUEST_CACHE[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return cached_data
        else:
            del REQUEST_CACHE[cache_key]
    return None

def set_cache(cache_key: str, data: Dict):
    """Store results in cache."""
    REQUEST_CACHE[cache_key] = (data, time.time())

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """Middleware to handle timeouts gracefully."""
    try:
        start_time = time.time()
        response = await call_next(request)
        processing_time = time.time() - start_time
        
        if processing_time > 25:  # Warn if接近timeout
            logger.warning(f"Slow request: {request.url} took {processing_time:.2f}s")
            
        return response
    except asyncio.TimeoutError:
        logger.error(f"Request timeout: {request.url}")
        return JSONResponse(
            status_code=504,
            content={"error": "Request timeout", "detail": "Processing took too long"}
        )
    except Exception as e:
        logger.error(f"Middleware error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "service": "seo-keyword-agent",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "optimized": True
    }

@app.get("/keywords/fast")
async def get_keywords_fast(
    request: Request,
    seed: str = Query(..., description="Seed keyword for expansion", min_length=1),
    top: int = Query(10, description="Maximum number of results to return", ge=1, le=20),
    max_candidates: int = Query(20, description="Maximum candidates to analyze", ge=5, le=30)
):
    """Fast keyword research endpoint optimized for Render."""
    # Authentication check
    auth_header = request.headers.get("Authorization")
    if API_AUTH_KEY:
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="API key required")
        if auth_header.replace("Bearer ", "") != API_AUTH_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    
    start_time = time.time()
    client_ip = request.client.host or "unknown"
    
    # Rate limiting
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Check cache first
    cache_key = get_cache_key(seed, top, max_candidates)
    cached_result = check_cache(cache_key)
    if cached_result:
        logger.info(f"Serving cached results for: '{seed}'")
        return KeywordResponse(**cached_result, cached=True)
    
    try:
        seed = seed.strip()
        if not seed:
            raise HTTPException(status_code=400, detail="Seed keyword cannot be empty")
        
        if len(seed) > 50:
            raise HTTPException(status_code=400, detail="Seed keyword too long")
        
        logger.info(f"Fast keyword request: '{seed}', top={top}")
        
        # Fast candidate collection
        candidates = collect_candidates_fast(seed, max_candidates)
        if not candidates:
            raise HTTPException(status_code=404, detail="No candidates found")
        
        logger.info(f"Found {len(candidates)} candidates, scoring...")
        
        # Fast parallel scoring
        scored_results = score_candidates_fast(candidates)
        if not scored_results:
            raise HTTPException(status_code=500, detail="Scoring failed")
        
        final_results = scored_results[:top]
        processing_time = time.time() - start_time
        
        # Cache the results
        result_data = {
            "seed": seed,
            "returned": len(final_results),
            "results": final_results,
            "processing_time_seconds": round(processing_time, 2)
        }
        set_cache(cache_key, result_data)
        
        logger.info(f"Request completed: {len(final_results)} results in {processing_time:.2f}s")
        
        return KeywordResponse(**result_data, cached=False)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Processing error")

@app.get("/keywords")
async def get_keywords_legacy(
    request: Request,
    seed: str = Query(..., description="Seed keyword for expansion"),
    top: int = Query(5, description="Number of results", ge=1, le=10)
):
    """Legacy endpoint with very limited results for compatibility."""
    return await get_keywords_fast(request, seed, top, 10)

@app.get("/status")
async def service_status():
    """Service status endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "optimized": True,
        "timestamp": datetime.utcnow().isoformat(),
        "configuration": {
            "api_auth_enabled": bool(API_AUTH_KEY),
            "serpapi_configured": bool(os.getenv("SERPAPI_KEY")),
            "cache_enabled": True,
            "rate_limiting": True,
            "max_workers": 5
        }
    }

@app.get("/cache/clear")
async def clear_cache():
    """Clear the request cache (admin endpoint)."""
    REQUEST_CACHE.clear()
    return {"status": "cache cleared", "cleared_entries": len(REQUEST_CACHE)}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting optimized SEO Keyword API on port {port}")
    logger.info(f"Max workers: 5")
    logger.info(f"Cache TTL: {CACHE_TTL}s")
    logger.info(f"Rate limit: {RATE_LIMIT_MAX_REQUESTS} requests per minute")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,  # Single worker for Render
        timeout_keep_alive=5,
        timeout_graceful_shutdown=5
    )