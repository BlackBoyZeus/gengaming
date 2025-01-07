// jwt-decode v3.1.2 - JWT token payload type definition
import { JwtPayload } from 'jwt-decode';

/**
 * User role enumeration for role-based access control
 */
export enum UserRole {
    ADMIN = 'ADMIN',
    DEVELOPER = 'DEVELOPER',
    USER = 'USER'
}

/**
 * Permission types for granular access control
 */
export enum Permission {
    GENERATE_CONTENT = 'GENERATE_CONTENT',
    CONTROL_CONTENT = 'CONTROL_CONTENT',
    TRAIN_MODELS = 'TRAIN_MODELS',
    CONFIGURE_SYSTEM = 'CONFIGURE_SYSTEM'
}

/**
 * Role hierarchy for permission comparison
 */
export const ROLE_HIERARCHY: Record<UserRole, number> = {
    [UserRole.ADMIN]: 3,
    [UserRole.DEVELOPER]: 2,
    [UserRole.USER]: 1
};

/**
 * Role-based permission mapping
 */
export const ROLE_PERMISSIONS: Record<UserRole, Permission[]> = {
    [UserRole.ADMIN]: [
        Permission.GENERATE_CONTENT,
        Permission.CONTROL_CONTENT,
        Permission.TRAIN_MODELS,
        Permission.CONFIGURE_SYSTEM
    ],
    [UserRole.DEVELOPER]: [
        Permission.GENERATE_CONTENT,
        Permission.CONTROL_CONTENT
    ],
    [UserRole.USER]: [
        Permission.GENERATE_CONTENT,
        Permission.CONTROL_CONTENT
    ]
};

/**
 * JWT token expiry time in seconds (1 hour)
 */
export const TOKEN_EXPIRY_TIME = 3600;

/**
 * User data interface with role and permissions
 */
export interface User {
    id: string;
    email: string;
    role: UserRole;
    permissions: Permission[];
}

/**
 * Authentication state interface with token management
 */
export interface AuthState {
    isAuthenticated: boolean;
    user: User | null;
    token: string | null;
    tokenExpiry: number | null;
}

/**
 * Login credentials interface for authentication
 */
export interface LoginCredentials {
    email: string;
    password: string;
}

/**
 * JWT token payload interface extending JwtPayload
 * with role and permissions
 */
export interface JWTToken extends JwtPayload {
    sub: string;
    role: UserRole;
    permissions: Permission[];
    exp: number;
    iat: number;
}