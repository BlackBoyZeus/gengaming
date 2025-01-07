"""
Core authentication module for GameGen-X API.
Implements JWT token handling, RBAC authorization, and user authentication.

FastAPI version: ^0.95.0
"""

# External imports with versions
from fastapi import HTTPException, Depends, Security  # ^0.95.0
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm  # ^0.95.0
from typing import Optional, Dict, Any
import logging
from datetime import timedelta

# Internal imports
from api.security.jwt import (
    create_access_token,
    decode_token,
    get_token_payload,
    JWTError
)
from api.security.rbac import (
    require_role,
    require_permission,
    RBACError
)
from db.models.user import User, UserRole
from core.exceptions import ValidationError
from core.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Configure OAuth2 password flow
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Constants
ACCESS_TOKEN_EXPIRE_MINUTES = 60
AUTH_ERROR_PREFIX = "AUTH"

class AuthError(ValidationError):
    """Custom exception for authentication-related errors with enhanced error tracking."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            validation_errors={"auth_error": str(original_error)} if original_error else {},
            validation_context="authentication"
        )
        self.message = message
        self.error_code = error_code or f"{AUTH_ERROR_PREFIX}-{message.replace(' ', '_').upper()}"
        
        # Log authentication error
        logger.error(
            f"Authentication Error: {message}",
            error_code=self.error_code,
            original_error=str(original_error) if original_error else None,
            exc_info=True
        )

async def authenticate_user(email: str, password: str) -> Optional[User]:
    """
    Authenticates a user with email and password using Argon2 hashing.
    
    Args:
        email: User's email address
        password: User's password
        
    Returns:
        Optional[User]: Authenticated user or None
        
    Raises:
        AuthError: If authentication fails
    """
    try:
        if not email or not password:
            raise AuthError(
                message="Email and password are required",
                error_code="AUTH_MISSING_CREDENTIALS"
            )

        # Get user by email (implementation would query database)
        user = User.get_by_email(email)  # Placeholder - actual DB query needed
        
        if not user:
            raise AuthError(
                message="Invalid email or password",
                error_code="AUTH_INVALID_CREDENTIALS"
            )
            
        if not user.is_active:
            raise AuthError(
                message="User account is inactive",
                error_code="AUTH_INACTIVE_ACCOUNT"
            )
            
        # Verify password using Argon2
        if not user.verify_password(password):
            raise AuthError(
                message="Invalid email or password",
                error_code="AUTH_INVALID_CREDENTIALS"
            )
            
        logger.info(
            f"User authenticated successfully",
            user_id=user.id,
            email=user.email
        )
        
        return user
        
    except Exception as e:
        if not isinstance(e, AuthError):
            raise AuthError(
                message="Authentication failed",
                error_code="AUTH_FAILED",
                original_error=e
            )
        raise

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    FastAPI dependency to get current authenticated user from JWT token.
    
    Args:
        token: JWT access token
        
    Returns:
        User: Current authenticated user
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    try:
        # Decode and validate JWT token
        payload = get_token_payload(token)
        
        if not payload:
            raise AuthError(
                message="Could not validate credentials",
                error_code="AUTH_INVALID_TOKEN"
            )
            
        # Get user from database
        user = User.get_by_id(payload["user_id"])  # Placeholder - actual DB query needed
        
        if not user:
            raise AuthError(
                message="User not found",
                error_code="AUTH_USER_NOT_FOUND"
            )
            
        if not user.is_active:
            raise AuthError(
                message="User is inactive",
                error_code="AUTH_INACTIVE_USER"
            )
            
        return user
        
    except (JWTError, AuthError) as e:
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )

async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()) -> Dict[str, Any]:
    """
    Endpoint handler for user login and JWT token generation.
    
    Args:
        form_data: OAuth2 password request form
        
    Returns:
        dict: Access token response
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Authenticate user
        user = await authenticate_user(form_data.username, form_data.password)
        
        if not user:
            raise AuthError(
                message="Incorrect email or password",
                error_code="AUTH_INVALID_CREDENTIALS"
            )
            
        # Generate access token
        access_token = create_access_token(
            user=user,
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        logger.info(
            f"Access token generated successfully",
            user_id=user.id,
            email=user.email
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user": {
                "id": user.id,
                "email": user.email,
                "role": user.role,
                "is_active": user.is_active
            }
        }
        
    except AuthError as e:
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Internal server error during authentication"
        )

# Export public interface
__all__ = [
    'authenticate_user',
    'get_current_user',
    'login_for_access_token',
    'AuthError'
]