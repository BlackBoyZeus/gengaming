"""
FastAPI routes initialization module that aggregates and exports all API route handlers
for the GameGen-X system, including video generation, control, status monitoring, and
WebSocket streaming endpoints.

FastAPI version: ^0.95.0
"""

# External imports with versions
from fastapi import APIRouter  # fastapi ^0.95.0

# Internal imports
from .generation import router as generation_router
from .control import router as control_router
from .status import router as status_router
from .websocket import router as websocket_router

# Global constants
API_PREFIX = '/api/v1'

# Initialize main router with prefix and tags
router = APIRouter(prefix=API_PREFIX, tags=['api'])

# Include sub-routers with appropriate prefixes and tags
router.include_router(
    generation_router,
    prefix='/generation',
    tags=['generation']
)

router.include_router(
    control_router,
    prefix='/control',
    tags=['control']
)

router.include_router(
    status_router,
    prefix='/status',
    tags=['status']
)

router.include_router(
    websocket_router,
    prefix='/ws',
    tags=['websocket']
)

# Export routers for application use
__all__ = [
    'router',
    'generation_router',
    'control_router', 
    'status_router',
    'websocket_router'
]