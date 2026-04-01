"""
NeuroAd Optimizer V2 - FastAPI Entry Point
==========================================
Production-ready backend for video cognitive analysis using TRIBE v2.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router
from app.services.patterns import initialize_pattern_engine
from app.services.tribe import initialize_tribe_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize services on startup."""
    logger.info("=" * 60)
    logger.info("NeuroAd Optimizer V2 - Starting up...")
    logger.info("=" * 60)
    
    start = time.perf_counter()
    
    # Initialize pattern engine (KMeans clustering)
    logger.info("Initializing pattern engine...")
    initialize_pattern_engine()
    logger.info("✓ Pattern engine ready (6 clusters)")
    
    # Initialize TRIBE v2 model (CRITICAL - no fallback)
    logger.info("Loading TRIBE v2 model (this may take a moment)...")
    initialize_tribe_model()
    logger.info("✓ TRIBE v2 model loaded successfully")
    
    elapsed = time.perf_counter() - start
    logger.info("=" * 60)
    logger.info(f"Server ready in {elapsed:.2f}s - POST /analyze available")
    logger.info("=" * 60)
    
    yield  # Server is running
    
    logger.info("Shutting down NeuroAd Optimizer V2...")


app = FastAPI(
    title="NeuroAd Optimizer V2",
    description="Video cognitive analysis using real TRIBE v2 neural encoding model",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "NeuroAd Optimizer V2"}
