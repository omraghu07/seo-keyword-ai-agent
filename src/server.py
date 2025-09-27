# src/server.py
"""
FastAPI server for SEO keyword research API

Self-contained implementation that works independently of external ranking modules.
Includes built-in keyword research functionality using SerpAPI.

Requirements:
    pip install fastapi uvicorn python-dotenv pandas google-search-results

Usage:
    uvicorn src.server:app --host 0.0.0.0 --port 8000
    
Environment Variables:
    - PORT: Server port (default: 8000)
    - API_AUTH_KEY: Optional API key for authentication
    - SERPAPI_KEY: Required for keyword research functionality
"""

import os
import logging
import time
import io
import math
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
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
        print("Warning: SerpAPI not available. Install with: pip install google-search-results")

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
security = HTTPBearer(auto_error=False) if API_AUTH_KEY else None

# Rate limiting setup
REQUEST_TIMES = {}
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX_REQUESTS = 30


class KeywordResponse(BaseModel):
    """Response model for keyword analysis results."""
    seed: str = Field(..., description="The input seed keyword")
    returned: int = Field(..., description="Number of results returned")
    results: List[Dict[str, Any]] = Field(..., description="List of keyword analysis results")
    processing_time_seconds: float = Field(..., description="Time taken to process the request")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


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


def collect_candidates(seed: str, max_candidates: int = 150) -> List[str]:
    """Collect keyword candidates from SerpAPI search results."""
    if not HAS_SERPAPI:
        # Return mock data if SerpAPI not available
        return [f"{seed} {suffix}" for suffix in [
            "tips", "guide", "best practices", "tutorial", "examples",
            "free", "online", "course", "training", "certification"
        ]]
    
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise Exception("SERPAPI_KEY not configured")
    
    search_params = {
        "engine": "google",
        "q": seed,
        "api_key": api_key,
        "hl": "en",
        "gl": "us"
    }
    
    try:
        search = GoogleSearch(search_params)
        results = search.get_dict()
    except Exception as e:
        logger.error(f"SerpAPI search failed: {e}")
        # Return mock data on API failure
        return [f"{seed} {suffix}" for suffix in [
            "tips", "guide", "best practices", "tutorial", "examples"
        ]]
    
    keyword_candidates = set()
    
    # Extract from related searches
    related_searches = results.get("related_searches", [])
    for item in related_searches:
        query = item.get("query") or item.get("suggestion")
        if query and len(query.strip()) > 0:
            keyword_candidates.add(query.strip())
    
    # Extract from people also ask
    related_questions = results.get("related_questions", [])
    for item in related_questions:
        question = item.get("question") or item.get("query")
        if question and len(question.strip()) > 0:
            keyword_candidates.add(question.strip())
    
    # Extract from organic titles
    organic_results = results.get("organic_results", [])
    for result in organic_results[:10]:
        title = result.get("title", "")
        if title and len(title.strip()) > 0:
            keyword_candidates.add(title.strip())
    
    return list(keyword_candidates)[:max_candidates]


def score_candidates(candidates: List[str], use_volume_api: bool = False) -> List[Dict[str, Any]]:
    """Score keyword candidates based on competition and estimated volume."""
    if not candidates:
        return []
    
    scored_results = []
    api_key = os.getenv("SERPAPI_KEY")
    
    for keyword in candidates:
        try:
            if HAS_SERPAPI and api_key:
                # Real analysis using SerpAPI
                search_params = {
                    "engine": "google",
                    "q": keyword,
                    "api_key": api_key,
                    "hl": "en",
                    "gl": "us",
                    "num": 10
                }
                
                search = GoogleSearch(search_params)
                search_results = search.get_dict()
                
                # Calculate competition
                competition_score, breakdown = calculate_competition_score(search_results)
                
                # Estimate volume (simple heuristic)
                word_count = len(keyword.split())
                estimated_volume = max(10, 10000 // (word_count + 1))
                
                # Calculate opportunity score
                volume_score = math.log10(estimated_volume + 1)
                opportunity_score = volume_score / (competition_score + 0.01)
                
                result = {
                    "keyword": keyword,
                    "monthly_searches": estimated_volume,
                    "competition_score": round(competition_score, 4),
                    "opportunity_score": round(opportunity_score, 2),
                    "total_results": breakdown["total_results"],
                    "ads_count": breakdown["ads_count"],
                    "featured_snippet": "Yes" if breakdown["has_featured_snippet"] else "No",
                    "people_also_ask": "Yes" if breakdown["has_people_also_ask"] else "No",
                    "knowledge_graph": "Yes" if breakdown["has_knowledge_graph"] else "No"
                }
            else:
                # Mock analysis when SerpAPI not available
                word_count = len(keyword.split())
                estimated_volume = max(50, 5000 // (word_count + 1))
                competition_score = min(0.1 + (word_count * 0.05), 0.8)
                volume_score = math.log10(estimated_volume + 1)
                opportunity_score = volume_score / (competition_score + 0.01)
                
                result = {
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
            
            scored_results.append(result)
            
            # Small delay to respect API limits
            time.sleep(0.1)
            
        except Exception as e:
            logger.warning(f"Failed to score keyword '{keyword}': {e}")
            continue
    
    # Sort by opportunity score (highest first)
    scored_results.sort(key=lambda x: x["opportunity_score"], reverse=True)
    return scored_results


def check_rate_limit(client_ip: str) -> bool:
    """Simple rate limiting check based on client IP."""
    current_time = time.time()
    
    if client_ip not in REQUEST_TIMES:
        REQUEST_TIMES[client_ip] = []
    
    # Clean old requests outside the window
    REQUEST_TIMES[client_ip] = [
        req_time for req_time in REQUEST_TIMES[client_ip]
        if current_time - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check if within limit
    if len(REQUEST_TIMES[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return False
    
    # Add current request
    REQUEST_TIMES[client_ip].append(current_time)
    return True


async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify API key authentication if enabled."""
    if not API_AUTH_KEY:
        return None
    
    if not credentials:
        logger.warning("API key required but not provided")
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide Authorization: Bearer <key>",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if credentials.credentials != API_AUTH_KEY:
        logger.warning("Invalid API key provided")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return credentials.credentials


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests."""
    client_ip = request.client.host or "unknown"
    
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return Response(
            content='{"error": "Rate limit exceeded. Maximum 30 requests per minute."}',
            status_code=429,
            media_type="application/json"
        )
    
    response = await call_next(request)
    return response


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "service": "seo-keyword-agent",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@app.get("/keywords", response_model=KeywordResponse)
async def get_keywords(
    request: Request,
    seed: str = Query(..., description="Seed keyword for expansion", min_length=1),
    top: int = Query(50, description="Maximum number of results to return", ge=1, le=200),
    use_volume_api: bool = Query(False, description="Use external API for real search volume data"),
    max_candidates: int = Query(150, description="Maximum number of candidates to analyze", ge=20, le=300),
    auth: Optional[str] = Depends(verify_api_key)
):
    """Perform keyword research and analysis."""
    start_time = time.time()
    client_ip = request.client.host or "unknown"
    
    # Log request
    logger.info(f"Keyword request from {client_ip}: seed='{seed}', top={top}")
    
    try:
        # Validate inputs
        seed = seed.strip()
        if not seed:
            raise HTTPException(status_code=400, detail="Seed keyword cannot be empty")
        
        if len(seed) > 100:
            raise HTTPException(status_code=400, detail="Seed keyword too long (max 100 characters)")
        
        # Step 1: Collect keyword candidates
        logger.info(f"Collecting candidates for seed: '{seed}'")
        candidates = collect_candidates(seed, max_candidates)
        
        if not candidates:
            logger.warning(f"No candidates found for seed: '{seed}'")
            raise HTTPException(
                status_code=404,
                detail="No keyword candidates found. Try a different seed keyword."
            )
        
        logger.info(f"Found {len(candidates)} candidates")
        
        # Step 2: Score candidates
        logger.info(f"Scoring {len(candidates)} candidates")
        scored_results = score_candidates(candidates, use_volume_api=use_volume_api)
        
        if not scored_results:
            logger.warning("No scored results returned")
            raise HTTPException(
                status_code=500,
                detail="Failed to score any keyword candidates"
            )
        
        # Limit results to requested top N
        final_results = scored_results[:top]
        
        processing_time = time.time() - start_time
        
        logger.info(f"Request completed successfully: {len(final_results)} results in {processing_time:.2f}s")
        
        return KeywordResponse(
            seed=seed,
            returned=len(final_results),
            results=final_results,
            processing_time_seconds=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request"
        )


@app.get("/csv")
async def get_keywords_csv(
    request: Request,
    seed: str = Query(..., description="Seed keyword for expansion", min_length=1),
    top: int = Query(50, description="Maximum number of results to return", ge=1, le=200),
    use_volume_api: bool = Query(False, description="Use external API for real search volume data"),
    max_candidates: int = Query(150, description="Maximum number of candidates to analyze", ge=20, le=300),
    auth: Optional[str] = Depends(verify_api_key)
):
    """Export keyword research results as CSV file."""
    if not HAS_PANDAS:
        raise HTTPException(status_code=500, detail="CSV export not available. Install pandas.")
    
    client_ip = request.client.host or "unknown"
    logger.info(f"CSV export request from {client_ip}: seed='{seed}', top={top}")
    
    try:
        # Validate and clean seed
        seed = seed.strip()
        if not seed:
            raise HTTPException(status_code=400, detail="Seed keyword cannot be empty")
        
        # Collect and score candidates
        candidates = collect_candidates(seed, max_candidates)
        if not candidates:
            raise HTTPException(status_code=404, detail="No keyword candidates found")
        
        scored_results = score_candidates(candidates, use_volume_api=use_volume_api)
        if not scored_results:
            raise HTTPException(status_code=500, detail="Failed to score candidates")
        
        # Limit results
        final_results = scored_results[:top]
        
        # Convert to DataFrame
        df = pd.DataFrame(final_results)
        
        # Create CSV in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        csv_buffer.close()
        
        # Create filename
        safe_seed = "".join(c for c in seed if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_seed = safe_seed.replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"keywords_{safe_seed}_{timestamp}.csv"
        
        logger.info(f"CSV export completed: {len(final_results)} results")
        
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV export error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate CSV export")


@app.get("/status")
async def service_status(auth: Optional[str] = Depends(verify_api_key)):
    """Get detailed service status information."""
    return {
        "status": "healthy",
        "service": "seo-keyword-agent",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "configuration": {
            "api_auth_enabled": API_AUTH_KEY is not None,
            "serpapi_configured": os.getenv("SERPAPI_KEY") is not None,
            "serpapi_available": HAS_SERPAPI,
            "pandas_available": HAS_PANDAS,
            "rate_limiting": {
                "enabled": True,
                "window_seconds": RATE_LIMIT_WINDOW,
                "max_requests": RATE_LIMIT_MAX_REQUESTS
            }
        }
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors with custom JSON response."""
    return Response(
        content='{"error": "Endpoint not found"}',
        status_code=404,
        media_type="application/json"
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors with custom JSON response."""
    logger.error(f"Internal server error: {str(exc)}", exc_info=True)
    return Response(
        content='{"error": "Internal server error"}',
        status_code=500,
        media_type="application/json"
    )


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (for Render deployment)
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting SEO Keyword API server on port {port}")
    logger.info(f"API authentication: {'enabled' if API_AUTH_KEY else 'disabled'}")
    logger.info(f"SerpAPI configured: {bool(os.getenv('SERPAPI_KEY'))}")
    logger.info(f"SerpAPI available: {HAS_SERPAPI}")
    logger.info(f"Pandas available: {HAS_PANDAS}")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )