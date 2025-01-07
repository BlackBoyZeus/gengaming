"""
Role-Based Access Control (RBAC) module for GameGen-X API security.
Implements hierarchical role management and permission validation with caching.

FastAPI version: ^0.95.0
cachetools version: ^5.0.0
"""

from typing import Callable, Dict, Optional, List
from fastapi import HTTPException, Depends, Security, Request
from cachetools import TTLCache, cached
import time

from db.models.user import User, UserRole
from core.exceptions import ValidationError
from core.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Role permissions matrix with inheritance
ROLE_PERMISSIONS: Dict[str, List[str]] = {
    'admin': ['generate_content', 'control_content', 'train_models', 'configure_system'],
    'developer': ['generate_content', 'control_content'],
    'user': ['generate_content', 'control_content']
}

# Role hierarchy levels (higher number = more privileges)
ROLE_HIERARCHY: Dict[str, int] = {
    'admin': 100,
    'developer': 50,
    'user': 10
}

# Cache configuration
PERMISSION_CACHE_TTL = 300  # 5 minutes
role_cache = TTLCache(maxsize=1000, ttl=PERMISSION_CACHE_TTL)
permission_cache = TTLCache(maxsize=1000, ttl=PERMISSION_CACHE_TTL)

class RBACError(ValidationError):
    """Custom exception for RBAC-related errors with enhanced error tracking."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        error_details: Dict
    ):
        super().__init__(
            message=message,
            validation_errors=error_details,
            validation_context="RBAC"
        )
        self.error_code = error_code
        self.error_details = error_details
        
        # Log RBAC error with full context
        logger.error(
            f"RBAC Error: {message}",
            error_code=error_code,
            error_details=error_details,
            exc_info=True
        )

    def to_dict(self) -> Dict:
        """Convert error to dictionary format with full context."""
        return {
            'error_code': self.error_code,
            'message': str(self),
            'details': self.error_details,
            'timestamp': time.time()
        }

@cached(cache=role_cache)
def check_role(user: User, required_role: UserRole) -> bool:
    """
    Checks if user has required role or higher in hierarchy with caching.
    
    Args:
        user: User instance to check
        required_role: Required role level
        
    Returns:
        bool: True if user has required role or higher
    """
    if not user or not user.role:
        return False
        
    user_level = ROLE_HIERARCHY.get(user.role, 0)
    required_level = ROLE_HIERARCHY.get(required_role.value, 0)
    
    return user_level >= required_level

@cached(cache=permission_cache)
def check_permission(user: User, required_permission: str) -> bool:
    """
    Checks if user has specific permission including inherited permissions.
    
    Args:
        user: User instance to check
        required_permission: Required permission name
        
    Returns:
        bool: True if user has permission
    """
    if not user or not user.role:
        return False
        
    role_permissions = ROLE_PERMISSIONS.get(user.role, [])
    return required_permission in role_permissions

def require_role(required_role: UserRole):
    """
    FastAPI dependency for role-based access control with audit logging.
    
    Args:
        required_role: Required role level
        
    Returns:
        Callable: FastAPI dependency function
    """
    async def role_checker(
        request: Request,
        user: User = Security(User)  # Assumes User model has security dependency
    ) -> User:
        logger.info(
            f"Checking role requirement",
            required_role=required_role.value,
            user_id=user.id,
            endpoint=request.url.path
        )
        
        if not check_role(user, required_role):
            error_details = {
                'user_role': user.role,
                'required_role': required_role.value,
                'user_id': user.id,
                'endpoint': request.url.path
            }
            raise HTTPException(
                status_code=403,
                detail=RBACError(
                    message=f"Insufficient role privileges",
                    error_code="RBAC_INSUFFICIENT_ROLE",
                    error_details=error_details
                ).to_dict()
            )
            
        logger.info(
            f"Role check passed",
            user_id=user.id,
            role=user.role,
            endpoint=request.url.path
        )
        return user
        
    return role_checker

def require_permission(required_permission: str):
    """
    FastAPI dependency for permission-based access control with inheritance.
    
    Args:
        required_permission: Required permission name
        
    Returns:
        Callable: FastAPI dependency function
    """
    async def permission_checker(
        request: Request,
        user: User = Security(User)  # Assumes User model has security dependency
    ) -> User:
        logger.info(
            f"Checking permission requirement",
            required_permission=required_permission,
            user_id=user.id,
            endpoint=request.url.path
        )
        
        if not check_permission(user, required_permission):
            error_details = {
                'user_role': user.role,
                'required_permission': required_permission,
                'user_id': user.id,
                'endpoint': request.url.path
            }
            raise HTTPException(
                status_code=403,
                detail=RBACError(
                    message=f"Insufficient permissions",
                    error_code="RBAC_INSUFFICIENT_PERMISSION",
                    error_details=error_details
                ).to_dict()
            )
            
        logger.info(
            f"Permission check passed",
            user_id=user.id,
            permission=required_permission,
            endpoint=request.url.path
        )
        return user
        
    return permission_checker

__all__ = [
    'require_role',
    'require_permission',
    'check_role',
    'check_permission',
    'RBACError'
]