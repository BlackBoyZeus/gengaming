/**
 * GameGen-X Authentication Context
 * @version 1.0.0
 * 
 * Implements secure authentication context with JWT token management,
 * role-based access control, and enhanced security features as specified
 * in Technical Specifications sections 7.1.1, 7.1.2, and 7.1.3
 */

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { AuthService } from '../services/auth';
import { AuthState, LoginCredentials, Permission, User } from '../types/auth';

// Constants for authentication management
const INITIAL_AUTH_STATE: AuthState = {
  isAuthenticated: false,
  user: null,
  token: null,
  tokenExpiry: null
};

const TOKEN_REFRESH_INTERVAL = 3300000; // 55 minutes in milliseconds
const RATE_LIMIT_ATTEMPTS = 5;
const RATE_LIMIT_WINDOW = 300000; // 5 minutes in milliseconds

/**
 * Interface defining the authentication context value type
 */
interface AuthContextType extends AuthState {
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => Promise<void>;
  checkAuth: () => Promise<void>;
  hasPermission: (permission: Permission) => Promise<boolean>;
  refreshToken: () => Promise<void>;
}

/**
 * Interface for AuthProvider props
 */
interface AuthProviderProps {
  children: React.ReactNode;
}

// Create the authentication context
const AuthContext = createContext<AuthContextType | null>(null);

/**
 * Authentication Provider Component with enhanced security features
 */
const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [state, setState] = useState<AuthState>(INITIAL_AUTH_STATE);
  const [refreshTimer, setRefreshTimer] = useState<NodeJS.Timeout | null>(null);
  const authService = new AuthService();

  // Setup automatic token refresh
  const setupRefreshTimer = useCallback(() => {
    if (refreshTimer) {
      clearInterval(refreshTimer);
    }
    const timer = setInterval(async () => {
      try {
        await authService.refreshToken();
        const newState = authService.getAuthState();
        setState(newState);
      } catch (error) {
        await logout();
      }
    }, TOKEN_REFRESH_INTERVAL);
    setRefreshTimer(timer);
  }, [refreshTimer]);

  // Enhanced login with rate limiting and security checks
  const login = async (credentials: LoginCredentials): Promise<void> => {
    try {
      const authState = await authService.login(credentials);
      setState(authState);
      setupRefreshTimer();
    } catch (error) {
      setState(INITIAL_AUTH_STATE);
      throw error;
    }
  };

  // Secure logout with token revocation
  const logout = async (): Promise<void> => {
    try {
      if (refreshTimer) {
        clearInterval(refreshTimer);
        setRefreshTimer(null);
      }
      await authService.revokeToken();
      authService.logout();
      setState(INITIAL_AUTH_STATE);
    } catch (error) {
      console.error('Logout error:', error);
      // Ensure state is cleared even if revocation fails
      setState(INITIAL_AUTH_STATE);
    }
  };

  // Enhanced authentication check with token validation
  const checkAuth = async (): Promise<void> => {
    try {
      const isValid = await authService.validateToken();
      if (!isValid) {
        await logout();
        return;
      }
      const currentUser = await authService.getCurrentUser();
      if (currentUser) {
        setState({
          isAuthenticated: true,
          user: currentUser,
          token: authService.getAuthState().token,
          tokenExpiry: authService.getAuthState().tokenExpiry
        });
        setupRefreshTimer();
      }
    } catch (error) {
      await logout();
    }
  };

  // Permission validation with caching
  const hasPermission = async (permission: Permission): Promise<boolean> => {
    return await authService.hasPermission(permission);
  };

  // Manual token refresh
  const refreshToken = async (): Promise<void> => {
    try {
      await authService.refreshToken();
      const newState = authService.getAuthState();
      setState(newState);
    } catch (error) {
      await logout();
    }
  };

  // Initial authentication check
  useEffect(() => {
    checkAuth();
    return () => {
      if (refreshTimer) {
        clearInterval(refreshTimer);
      }
    };
  }, []);

  const contextValue: AuthContextType = {
    ...state,
    login,
    logout,
    checkAuth,
    hasPermission,
    refreshToken
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

/**
 * Custom hook for accessing authentication context
 */
const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export { AuthContext, AuthProvider, useAuth };
export type { AuthContextType, AuthProviderProps };