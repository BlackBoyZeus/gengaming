/**
 * GameGen-X Authentication Hook
 * @version 1.0.0
 * 
 * Custom React hook implementing secure JWT-based authentication and role-based access control
 * as specified in Technical Specifications sections 7.1.1, 7.1.2, and 7.1.3
 */

import { useContext } from 'react'; // v18.0.0
import jwtDecode from 'jwt-decode'; // v3.1.2
import { AuthContext } from '../contexts/AuthContext';
import { AuthState, Permission } from '../types/auth';

// Cache duration for permission checks (5 minutes)
const PERMISSION_CACHE_DURATION = 300000;

// Rate limiting configuration for auth operations
const AUTH_RATE_LIMIT = {
  MAX_ATTEMPTS: 5,
  WINDOW_MS: 300000, // 5 minutes
  RESET_AFTER_MS: 3600000 // 1 hour
};

/**
 * Interface for enhanced authentication hook return type
 */
interface UseAuthReturn {
  // Auth state
  authState: AuthState;
  // Core auth methods
  login: (credentials: { email: string; password: string }) => Promise<void>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<void>;
  // Permission validation
  checkPermission: (permission: Permission) => boolean;
  validateSession: () => Promise<boolean>;
  // Enhanced security methods
  validateTokenExpiry: () => boolean;
  clearAuthData: () => void;
}

/**
 * Permission check cache implementation
 */
class PermissionCache {
  private cache: Map<string, { result: boolean; timestamp: number }> = new Map();

  set(key: string, value: boolean): void {
    this.cache.set(key, {
      result: value,
      timestamp: Date.now()
    });
  }

  get(key: string): boolean | null {
    const cached = this.cache.get(key);
    if (!cached) return null;

    if (Date.now() - cached.timestamp > PERMISSION_CACHE_DURATION) {
      this.cache.delete(key);
      return null;
    }

    return cached.result;
  }

  clear(): void {
    this.cache.clear();
  }
}

// Initialize permission cache
const permissionCache = new PermissionCache();

/**
 * Custom hook for secure authentication and authorization management
 */
export const useAuth = (): UseAuthReturn => {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }

  /**
   * Validates JWT token expiration
   */
  const validateTokenExpiry = (): boolean => {
    const { token } = context;
    if (!token) return false;

    try {
      const decoded = jwtDecode<{ exp: number }>(token);
      return Date.now() < (decoded.exp * 1000);
    } catch {
      return false;
    }
  };

  /**
   * Enhanced permission check with caching
   */
  const checkPermission = (permission: Permission): boolean => {
    const { user, isAuthenticated } = context;
    
    if (!isAuthenticated || !user) {
      return false;
    }

    const cacheKey = `${user.id}:${permission}`;
    const cachedResult = permissionCache.get(cacheKey);
    
    if (cachedResult !== null) {
      return cachedResult;
    }

    const hasPermission = user.permissions.includes(permission);
    permissionCache.set(cacheKey, hasPermission);
    
    return hasPermission;
  };

  /**
   * Validates current session status
   */
  const validateSession = async (): Promise<boolean> => {
    if (!validateTokenExpiry()) {
      try {
        await context.refreshToken();
        return true;
      } catch {
        await context.logout();
        return false;
      }
    }
    return true;
  };

  /**
   * Securely clears all authentication data
   */
  const clearAuthData = (): void => {
    permissionCache.clear();
    context.logout();
  };

  return {
    // Expose auth context state
    authState: {
      isAuthenticated: context.isAuthenticated,
      user: context.user,
      token: context.token,
      tokenExpiry: context.tokenExpiry
    },
    // Core auth methods with enhanced security
    login: context.login,
    logout: async () => {
      await context.logout();
      clearAuthData();
    },
    refreshToken: context.refreshToken,
    // Permission and session validation
    checkPermission,
    validateSession,
    // Enhanced security methods
    validateTokenExpiry,
    clearAuthData
  };
};

/**
 * Hook for simplified permission checking
 */
export const usePermission = (permission: Permission): boolean => {
  const { checkPermission } = useAuth();
  return checkPermission(permission);
};

export type { UseAuthReturn };