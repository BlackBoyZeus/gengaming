# External imports with versions
from pathlib import Path  # pathlib ^3.9
from python_jose import jwt  # python-jose[cryptography] ^3.3.0
from typing import Dict, List, Optional, Union, Callable  # typing ^3.9
from datetime import datetime, timedelta
from functools import wraps
import logging

# Internal imports
from ...core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# JWT Configuration
ALGORITHM = "RS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Key file paths
PRIVATE_KEY_PATH = Path('keys/private.pem')
PUBLIC_KEY_PATH = Path('keys/public.pem')

# Role definitions
ROLE_ADMIN = "admin"
ROLE_DEVELOPER = "developer"
ROLE_USER = "user"

# Permission definitions
PERM_GENERATE = "generate"
PERM_CONTROL = "control"
PERM_TRAIN = "train"
PERM_CONFIGURE = "configure"

# Role-Permission mapping
ROLE_PERMISSIONS: Dict[str, List[str]] = {
    ROLE_ADMIN: [PERM_GENERATE, PERM_CONTROL, PERM_TRAIN, PERM_CONFIGURE],
    ROLE_DEVELOPER: [PERM_GENERATE, PERM_CONTROL],
    ROLE_USER: [PERM_GENERATE, PERM_CONTROL]
}

def create_access_token(
    user_id: str,
    role: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token with role and permissions.
    
    Args:
        user_id: Unique identifier for the user
        role: User's role (admin, developer, user)
        expires_delta: Optional custom expiration time
        
    Returns:
        str: Encoded JWT token
    """
    if role not in ROLE_PERMISSIONS:
        raise ValueError(f"Invalid role: {role}")
        
    expire = datetime.utcnow() + (
        expires_delta if expires_delta
        else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    token_data = {
        "sub": user_id,
        "role": role,
        "permissions": ROLE_PERMISSIONS[role],
        "exp": expire,
        "env": settings.environment
    }
    
    with open(PRIVATE_KEY_PATH, 'rb') as key_file:
        private_key = key_file.read()
        
    try:
        encoded_token = jwt.encode(
            token_data,
            private_key,
            algorithm=ALGORITHM
        )
        return encoded_token
    except Exception as e:
        logger.error(f"Token creation failed: {str(e)}")
        raise

def verify_token(token: str) -> Dict:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Dict: Decoded token payload
    """
    try:
        with open(PUBLIC_KEY_PATH, 'rb') as key_file:
            public_key = key_file.read()
            
        payload = jwt.decode(
            token,
            public_key,
            algorithms=[ALGORITHM]
        )
        
        if payload["env"] != settings.environment:
            raise ValueError("Token environment mismatch")
            
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        raise
    except jwt.JWTError as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Token processing error: {str(e)}")
        raise

def verify_role(token_data: Dict, required_role: str) -> bool:
    """
    Verify if user has required role.
    
    Args:
        token_data: Decoded token payload
        required_role: Role to check for
        
    Returns:
        bool: True if user has required role
    """
    user_role = token_data.get("role")
    if not user_role:
        return False
        
    # Admin role has access to everything
    if user_role == ROLE_ADMIN:
        return True
        
    return user_role == required_role

def require_role(required_role: str) -> Callable:
    """
    Decorator for role-based endpoint protection.
    
    Args:
        required_role: Role required to access endpoint
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            token_data = kwargs.get("token_data")
            if not token_data:
                raise ValueError("No token data provided")
                
            if not verify_role(token_data, required_role):
                raise ValueError(f"Role {required_role} required")
                
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def check_permission(token_data: Dict, required_permission: str) -> bool:
    """
    Check if user has required permission.
    
    Args:
        token_data: Decoded token payload
        required_permission: Permission to check for
        
    Returns:
        bool: True if user has required permission
    """
    user_permissions = token_data.get("permissions", [])
    return required_permission in user_permissions

def require_permission(required_permission: str) -> Callable:
    """
    Decorator for permission-based endpoint protection.
    
    Args:
        required_permission: Permission required to access endpoint
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            token_data = kwargs.get("token_data")
            if not token_data:
                raise ValueError("No token data provided")
                
            if not check_permission(token_data, required_permission):
                raise ValueError(f"Permission {required_permission} required")
                
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Export security components
__all__ = [
    "ALGORITHM",
    "ACCESS_TOKEN_EXPIRE_MINUTES",
    "ROLE_ADMIN",
    "ROLE_DEVELOPER", 
    "ROLE_USER",
    "PERM_GENERATE",
    "PERM_CONTROL",
    "PERM_TRAIN",
    "PERM_CONFIGURE",
    "create_access_token",
    "verify_token",
    "verify_role",
    "require_role",
    "check_permission",
    "require_permission"
]