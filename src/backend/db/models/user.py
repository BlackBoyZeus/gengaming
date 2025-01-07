"""
SQLAlchemy model for user management in GameGen-X.
Implements secure user data storage, role-based access control, and authentication with Argon2 hashing.

SQLAlchemy version: ^1.4.41
Passlib version: ^1.7.4
Email-validator version: ^2.0.0
"""

from enum import Enum
from sqlalchemy import Column, String, Boolean, Integer, Index, UniqueConstraint
from sqlalchemy.orm import relationship, validates
from passlib.hash import argon2
from email_validator import validate_email as validate_email_format, EmailNotValidError

from ..base import Base, BaseModel

# User role enumeration
class UserRole(str, Enum):
    admin = "admin"
    developer = "developer"
    user = "user"

# Argon2 configuration for secure password hashing
ARGON2_TIME_COST = 4
ARGON2_MEMORY_COST = 65536  # 64MB
ARGON2_PARALLELISM = 2
MIN_PASSWORD_LENGTH = 12

class User(BaseModel):
    """
    User model for authentication and authorization with secure password handling
    and role-based access control.
    """
    __tablename__ = 'users'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # User identification and authentication
    email = Column(String(255), nullable=False, unique=True, index=True)
    hashed_password = Column(String(1024), nullable=False)
    full_name = Column(String(255), nullable=False)
    
    # Account status and role
    is_active = Column(Boolean, nullable=False, default=True)
    role = Column(
        String(20),
        nullable=False,
        default=UserRole.user.value
    )

    # Relationships
    generations = relationship("Generation", back_populates="user", cascade="all, delete-orphan")

    # Indexes and constraints
    __table_args__ = (
        Index('ix_users_email_role', 'email', 'role'),
        UniqueConstraint('email', name='uq_users_email'),
    )

    def __init__(self, email: str, password: str, full_name: str, 
                 role: UserRole = UserRole.user, is_active: bool = True):
        """
        Initialize user with secure password hashing and validation.

        Args:
            email: User's email address
            password: Plain text password to be hashed
            full_name: User's full name
            role: User's role (default: user)
            is_active: Account status (default: True)
        """
        super().__init__()
        
        # Validate and set email
        self.email = self.validate_email(email)
        
        # Validate and hash password
        if self.validate_password(password):
            self.hashed_password = argon2.using(
                time_cost=ARGON2_TIME_COST,
                memory_cost=ARGON2_MEMORY_COST,
                parallelism=ARGON2_PARALLELISM
            ).hash(password)
        
        self.full_name = full_name.strip()
        self.role = role.value if isinstance(role, UserRole) else role
        self.is_active = is_active

    @validates('email')
    def validate_email(self, email: str) -> str:
        """
        Validate email format and normalize.

        Args:
            email: Email address to validate

        Returns:
            Normalized email address

        Raises:
            ValueError: If email is invalid
        """
        if not email:
            raise ValueError("Email is required")
            
        try:
            # Validate and normalize email
            email_info = validate_email_format(email, check_deliverability=False)
            return email_info.normalized
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email format: {str(e)}")

    @validates('password')
    def validate_password(self, password: str) -> bool:
        """
        Validate password strength requirements.

        Args:
            password: Password to validate

        Returns:
            True if password meets requirements

        Raises:
            ValueError: If password is invalid
        """
        if not password or len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long")

        # Check for character complexity
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)

        if not all([has_upper, has_lower, has_digit, has_special]):
            raise ValueError("Password must contain uppercase, lowercase, digit, and special characters")

        return True

    def verify_password(self, password: str) -> bool:
        """
        Verify password against stored hash.

        Args:
            password: Password to verify

        Returns:
            True if password matches, False otherwise
        """
        if not password:
            return False
            
        return argon2.verify(password, self.hashed_password)

    def to_dict(self) -> dict:
        """
        Convert user to dictionary excluding sensitive data.

        Returns:
            Dictionary with user data
        """
        data = super().to_dict()
        
        # Remove sensitive fields
        data.pop('hashed_password', None)
        
        # Add role information
        data['role'] = self.role
        data['is_active'] = self.is_active
        
        return data