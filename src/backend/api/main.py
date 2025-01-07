# External imports with versions
from fastapi import FastAPI, Request, Response  # fastapi ^0.95.0
from fastapi.middleware.cors import CORSMiddleware  # fastapi ^0.95.0
import uvicorn  # uvicorn ^0.21.0
from prometheus_client import start_http_server  # prometheus_client ^0.16.0
from contextlib import asynccontextmanager
import logging
import time

# Internal imports
from api.config import api_settings, security_settings, websocket_settings
from api.middleware import setup_middleware
from api.routes.generation import router as generation_router
from api.routes.control import router as control_router

# Initialize logger
logger = logging.getLogger(__name__)

@asynccontextmanager
async def create_application():
    """Creates and configures the FastAPI application with comprehensive security and monitoring."""
    try:
        # Initialize FastAPI with FreeBSD-optimized settings
        app = FastAPI(
            title=api_settings.title,
            version=api_settings.version,
            **api_settings.freebsd_worker_config
        )

        # Configure CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=api_settings.cors_origins.get(api_settings.environment, ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Process-Time", "X-Request-ID"]
        )

        # Setup comprehensive middleware stack
        setup_middleware(app)

        # Include routers
        app.include_router(generation_router)
        app.include_router(control_router)

        # Configure WebSocket handlers
        await configure_websocket(app)

        # Setup shutdown handlers
        await setup_shutdown_handlers(app)

        # Health check endpoint
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": api_settings.version
            }

        # Request timing middleware
        @app.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response

        yield app

    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise

async def configure_websocket(app: FastAPI):
    """Configures WebSocket handlers with optimized settings."""
    # Configure WebSocket connection pool
    app.websocket_pool = {
        "max_size": websocket_settings.connection_pool_config["max_size"],
        "connections": set(),
        "performance": websocket_settings.performance_tuning
    }

    # Configure WebSocket options
    app.websocket_options = {
        "ping_interval": websocket_settings.ping_interval,
        "ping_timeout": websocket_settings.ping_timeout,
        "close_timeout": websocket_settings.close_timeout,
        "max_message_size": websocket_settings.max_message_size,
        "socket_options": websocket_settings.freebsd_socket_opts
    }

async def setup_shutdown_handlers(app: FastAPI):
    """Configures graceful shutdown handlers."""
    @app.on_event("shutdown")
    async def shutdown_event():
        # Close all WebSocket connections
        for connection in app.websocket_pool["connections"]:
            await connection.close(code=1000)

        # Clear caches
        app.websocket_pool["connections"].clear()

        logger.info("Application shutdown completed")

def main():
    """Entry point for running the FastAPI application."""
    try:
        # Start metrics server
        start_http_server(9090)

        # Configure uvicorn with FreeBSD optimizations
        uvicorn_config = {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": api_settings.workers,
            "loop": "uvloop",
            "http": "httptools",
            "ws": "websockets",
            "log_level": "info",
            "access_log": True,
            "proxy_headers": True,
            "forwarded_allow_ips": "*",
            "server_header": False,
            "date_header": True,
            "timeout_keep_alive": 30,
            **api_settings.freebsd_worker_config
        }

        # Create and run application
        app = FastAPI()
        uvicorn.run(
            "main:app",
            **uvicorn_config
        )

    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()