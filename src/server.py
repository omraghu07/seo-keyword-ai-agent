# src/server.py
"""
FastAPI server for SEO keyword research API

Provides REST endpoints for keyword expansion and scoring functionality.
Designed for n8n and automation workflows.

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
import sys
from pathlib import Path
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
except ImportError:
    pd = None

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import ranking functions - handle different import scenarios
try:
    from src.ranking import collect_candidates, score_candidates
except ImportError:
    try:
        import ranking
        collect_candidates = ranking.collect_candidates
        score_candidates = ranking.score_candidates
    except ImportError as e:
        print(f"Error importing ranking functions: {e}")
        # Create placeholder functions for testing
        def collect_candidates(seed: str, max_candidates: int = 150) -> List[str]:
            return [f"{seed} keyword {i}" for i in range(1, min(max_candidates + 1, 11))]
        
        def score_candidates(candidates: List[str], use_volume_api: bool = False) -> List[Dict[str, Any]]:
            return [
                {
                    "keyword": keyword,
                    "monthly_searches": 1000 - i * 50,
                    "competition_score": 0.1 + (i * 0.05),
                    "opportunity_score": 100 - i * 5,
                    "total_results": 1000000 - i * 50000,
                    "ads_count": i % 3,
                    "has_featured_snippet": i % 2 == 0,
                    "has_people_also_ask": i % 3 == 0,
                    "has_knowledge_graph": i % 4 == 0
                }
                for i, keyword in enumerate(candidates, 1)
            ]

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
    """
    Perform keyword research and analysis.
    
    This endpoint expands a seed keyword into related terms, analyzes competition
    and search volume, then returns scored opportunities ranked by potential value.
    """
    start_time = time.time()
    client_ip = request.client.host or "unknown"
    
    # Log request
    logger.info(f"Keyword request from {client_ip}: seed='{seed}', top={top}, "
                f"use_volume_api={use_volume_api}, auth_provided={auth is not None}")
    
    try:
        # Validate environment (only check if not using placeholder functions)
        if 'ranking' in sys.modules and not os.getenv("SERPAPI_KEY"):
            logger.error("SERPAPI_KEY not configured")
            raise HTTPException(
                status_code=500,
                detail="Service not properly configured. SERPAPI_KEY missing."
            )
        
        # Validate inputs
        seed = seed.strip()
        if not seed:
            raise HTTPException(status_code=400, detail="Seed keyword cannot be empty")
        
        if len(seed) > 100:
            raise HTTPException(status_code=400, detail="Seed keyword too long (max 100 characters)")
        
        # Step 1: Collect keyword candidates
        logger.info(f"Collecting candidates for seed: '{seed}'")
        try:
            candidates = collect_candidates(seed, max_candidates)
        except Exception as e:
            logger.error(f"Failed to collect candidates: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to collect keyword candidates: {str(e)}"
            )
        
        if not candidates:
            logger.warning(f"No candidates found for seed: '{seed}'")
            raise HTTPException(
                status_code=404,
                detail="No keyword candidates found. Try a different seed keyword."
            )
        
        logger.info(f"Found {len(candidates)} candidates")
        
        # Step 2: Score candidates
        logger.info(f"Scoring {len(candidates)} candidates")
        try:
            scored_results = score_candidates(candidates, use_volume_api=use_volume_api)
        except Exception as e:
            logger.error(f"Failed to score candidates: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to score keyword candidates: {str(e)}"
            )
        
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
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
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
    if not pd:
        raise HTTPException(status_code=500, detail="CSV export not available. Install pandas.")
    
    client_ip = request.client.host or "unknown"
    logger.info(f"CSV export request from {client_ip}: seed='{seed}', top={top}")
    
    try:
        # Validate environment
        if 'ranking' in sys.modules and not os.getenv("SERPAPI_KEY"):
            raise HTTPException(status_code=500, detail="Service not properly configured")
        
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
            "pandas_available": pd is not None,
            "ranking_module_loaded": 'ranking' in sys.modules,
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
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )