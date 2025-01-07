# External imports with versions
from jose import jwt, JWTError as JoseJWTError  # python-jose ^3.3.0
from datetime import datetime, timedelta, UTC  # ^3.9
from typing import Optional, Dict, Any  # ^3.9
import redis  # redis ^4.5.0
import uuid
import logging

# Internal imports
from core.config import Settings
from core.exceptions import ValidationError
from db.models.user import User

# Configure logging
logger = logging.getLogger(__name__)

# Constants for JWT configuration
ALGORITHM = "RS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
TOKEN_TYPE = "bearer"
REVOCATION_KEY_PREFIX = "revoked_token:"

# Initialize Redis connection for token revocation
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

class JWTError(ValidationError):
    """Enhanced custom exception for JWT-related errors with detailed error information."""
    
    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        error_code: Optional[str] = None,
        recovery_hint: Optional[str] = None
    ):
        """Initialize JWT error with comprehensive error details."""
        super().__init__(
            message=message,
            validation_errors={"jwt_error": str(original_error)} if original_error else {},
            validation_context="jwt_validation"
        )
        self.message = message
        self.original_error = original_error
        self.error_code = error_code or f"JWT-{str(uuid.uuid4())[:8]}"
        self.recovery_hint = recovery_hint

    def to_dict(self) -> Dict[str, Any]:
        """Converts error to dictionary format."""
        error_dict = {
            "error": self.message,
            "error_code": self.error_code,
            "type": "jwt_error"
        }
        if self.original_error:
            error_dict["original_error"] = str(self.original_error)
        if self.recovery_hint:
            error_dict["recovery_hint"] = self.recovery_hint
        return error_dict

def create_access_token(
    user: User,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Creates a new JWT access token for a user with enhanced security metadata.
    
    Args:
        user: User instance to generate token for
        expires_delta: Optional custom expiration time
        
    Returns:
        str: Encoded JWT token
        
    Raises:
        JWTError: If token creation fails
    """
    try:
        # Verify user is active
        if not user.is_active:
            raise JWTError(
                message="Cannot create token for inactive user",
                error_code="JWT_INACTIVE_USER",
                recovery_hint="Contact administrator to activate account"
            )

        # Calculate expiration time
        expire = datetime.now(UTC) + (
            expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )

        # Create token payload with security metadata
        token_data = {
            "sub": str(user.id),
            "role": user.role,
            "type": TOKEN_TYPE,
            "exp": expire,
            "iat": datetime.now(UTC),
            "jti": str(uuid.uuid4()),
            "env": Settings().environment
        }

        # Sign token with RS256 using private key
        encoded_token = jwt.encode(
            token_data,
            Settings().jwt_private_key,
            algorithm=ALGORITHM
        )

        logger.info(
            f"Created access token for user {user.id}",
            extra={"user_id": user.id, "token_jti": token_data["jti"]}
        )

        return encoded_token

    except Exception as e:
        raise JWTError(
            message="Failed to create access token",
            original_error=e,
            error_code="JWT_CREATION_ERROR"
        )

def decode_token(token: str) -> Dict[str, Any]:
    """
    Decodes and validates a JWT token with comprehensive security checks.
    
    Args:
        token: JWT token to decode
        
    Returns:
        dict: Decoded token payload
        
    Raises:
        JWTError: If token is invalid or verification fails
    """
    try:
        # Check token format
        if not token or not isinstance(token, str):
            raise JWTError(
                message="Invalid token format",
                error_code="JWT_INVALID_FORMAT"
            )

        # Check revocation status
        if redis_client.exists(f"{REVOCATION_KEY_PREFIX}{token}"):
            raise JWTError(
                message="Token has been revoked",
                error_code="JWT_REVOKED",
                recovery_hint="Please obtain a new token"
            )

        # Verify token signature and decode payload
        payload = jwt.decode(
            token,
            Settings().jwt_public_key,
            algorithms=[ALGORITHM]
        )

        # Validate token type and environment
        if payload.get("type") != TOKEN_TYPE:
            raise JWTError(
                message="Invalid token type",
                error_code="JWT_INVALID_TYPE"
            )

        if payload.get("env") != Settings().environment:
            raise JWTError(
                message="Token from different environment",
                error_code="JWT_ENV_MISMATCH"
            )

        return payload

    except JoseJWTError as e:
        raise JWTError(
            message="Failed to decode token",
            original_error=e,
            error_code="JWT_DECODE_ERROR",
            recovery_hint="Token may be expired or invalid"
        )
    except Exception as e:
        raise JWTError(
            message="Token validation failed",
            original_error=e,
            error_code="JWT_VALIDATION_ERROR"
        )

def get_token_payload(token: str) -> Dict[str, Any]:
    """
    Extracts and validates user information from token.
    
    Args:
        token: JWT token to process
        
    Returns:
        dict: User data and metadata from token
        
    Raises:
        JWTError: If token or user validation fails
    """
    try:
        # Decode and validate token
        payload = decode_token(token)

        # Extract user data
        user_data = {
            "user_id": payload["sub"],
            "role": payload["role"],
            "token_type": payload["type"],
            "expires_at": datetime.fromtimestamp(payload["exp"], UTC).isoformat(),
            "issued_at": datetime.fromtimestamp(payload["iat"], UTC).isoformat(),
            "token_id": payload["jti"],
            "environment": payload["env"]
        }

        return user_data

    except Exception as e:
        raise JWTError(
            message="Failed to extract token payload",
            original_error=e,
            error_code="JWT_PAYLOAD_ERROR"
        )

def revoke_token(token: str) -> bool:
    """
    Adds a token to the revocation list.
    
    Args:
        token: JWT token to revoke
        
    Returns:
        bool: Success status
        
    Raises:
        JWTError: If token revocation fails
    """
    try:
        # Decode token to get expiry
        payload = decode_token(token)
        expiry = datetime.fromtimestamp(payload["exp"], UTC) - datetime.now(UTC)

        # Add to revocation list with expiry
        revocation_key = f"{REVOCATION_KEY_PREFIX}{token}"
        redis_client.setex(
            revocation_key,
            int(expiry.total_seconds()),
            "revoked"
        )

        logger.info(
            f"Token revoked successfully",
            extra={"token_jti": payload["jti"]}
        )

        return True

    except Exception as e:
        raise JWTError(
            message="Failed to revoke token",
            original_error=e,
            error_code="JWT_REVOCATION_ERROR"
        )