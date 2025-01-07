/**
 * GameGen-X Authentication Service
 * @version 1.0.0
 * 
 * Implements secure JWT token-based authentication and role-based access control
 * as specified in Technical Specifications sections 7.1.1, 7.1.2, and 7.1.3
 */

import jwtDecode from 'jwt-decode'; // v3.1.2
import axios from 'axios'; // v1.4.0
import CryptoJS from 'crypto-js'; // v4.1.1

import { API_CONFIG } from '../config/api';
import { 
    User, 
    UserRole, 
    Permission, 
    LoginCredentials, 
    AuthState, 
    JWTToken, 
    ROLE_PERMISSIONS, 
    TOKEN_EXPIRY_TIME 
} from '../types/auth';

// Constants for token management and rate limiting
const TOKEN_STORAGE_KEY = 'gamegen_x_auth_token';
const TOKEN_EXPIRY_BUFFER = 300; // 5 minutes before expiry
const MAX_LOGIN_ATTEMPTS = 5;
const LOGIN_ATTEMPT_WINDOW = 300000; // 5 minutes in milliseconds
const ENCRYPTION_KEY = process.env.VITE_TOKEN_ENCRYPTION_KEY || 'default-key';

/**
 * Class managing secure token storage with encryption
 */
class TokenManager {
    private encryptToken(token: string): string {
        return CryptoJS.AES.encrypt(token, ENCRYPTION_KEY).toString();
    }

    private decryptToken(encryptedToken: string): string {
        const bytes = CryptoJS.AES.decrypt(encryptedToken, ENCRYPTION_KEY);
        return bytes.toString(CryptoJS.enc.Utf8);
    }

    public storeToken(token: string): void {
        const encryptedToken = this.encryptToken(token);
        localStorage.setItem(TOKEN_STORAGE_KEY, encryptedToken);
    }

    public getToken(): string | null {
        const encryptedToken = localStorage.getItem(TOKEN_STORAGE_KEY);
        if (!encryptedToken) return null;
        return this.decryptToken(encryptedToken);
    }

    public removeToken(): void {
        localStorage.removeItem(TOKEN_STORAGE_KEY);
    }
}

/**
 * Class managing user session state
 */
class SessionManager {
    private currentUser: User | null = null;
    private tokenExpiry: number | null = null;

    public setSession(token: string, user: User): void {
        this.currentUser = user;
        const decoded = jwtDecode<JWTToken>(token);
        this.tokenExpiry = decoded.exp;
    }

    public clearSession(): void {
        this.currentUser = null;
        this.tokenExpiry = null;
    }

    public getCurrentUser(): User | null {
        return this.currentUser;
    }

    public isTokenExpired(): boolean {
        if (!this.tokenExpiry) return true;
        return (Date.now() / 1000) >= (this.tokenExpiry - TOKEN_EXPIRY_BUFFER);
    }
}

/**
 * Class implementing rate limiting for login attempts
 */
class RateLimiter {
    private attempts: Map<string, number[]> = new Map();

    public checkRateLimit(email: string): boolean {
        const now = Date.now();
        const attempts = this.attempts.get(email) || [];
        
        // Remove expired attempts
        const validAttempts = attempts.filter(time => 
            now - time < LOGIN_ATTEMPT_WINDOW
        );
        
        if (validAttempts.length >= MAX_LOGIN_ATTEMPTS) {
            return false;
        }

        validAttempts.push(now);
        this.attempts.set(email, validAttempts);
        return true;
    }

    public resetAttempts(email: string): void {
        this.attempts.delete(email);
    }
}

/**
 * Main authentication service class implementing secure auth functionality
 */
export class AuthService {
    private tokenManager: TokenManager;
    private sessionManager: SessionManager;
    private rateLimiter: RateLimiter;

    constructor() {
        this.tokenManager = new TokenManager();
        this.sessionManager = new SessionManager();
        this.rateLimiter = new RateLimiter();
        this.setupAxiosInterceptors();
    }

    /**
     * Sets up axios interceptors for automatic token handling
     */
    private setupAxiosInterceptors(): void {
        axios.interceptors.request.use(
            (config) => {
                const token = this.tokenManager.getToken();
                if (token) {
                    config.headers.Authorization = `Bearer ${token}`;
                }
                return config;
            },
            (error) => Promise.reject(error)
        );

        axios.interceptors.response.use(
            (response) => response,
            async (error) => {
                if (error.response?.status === 401 && this.tokenManager.getToken()) {
                    try {
                        await this.refreshToken();
                        return axios(error.config);
                    } catch (refreshError) {
                        this.logout();
                        throw refreshError;
                    }
                }
                throw error;
            }
        );
    }

    /**
     * Authenticates user with credentials
     */
    public async login(credentials: LoginCredentials): Promise<AuthState> {
        if (!this.rateLimiter.checkRateLimit(credentials.email)) {
            throw new Error('Too many login attempts. Please try again later.');
        }

        try {
            const response = await axios.post(
                `${API_CONFIG.BASE_URL}/api/auth/login`,
                credentials
            );

            const { token, user } = response.data;
            this.tokenManager.storeToken(token);
            this.sessionManager.setSession(token, user);
            this.rateLimiter.resetAttempts(credentials.email);

            return {
                isAuthenticated: true,
                user,
                token,
                tokenExpiry: this.sessionManager.isTokenExpired() ? null : TOKEN_EXPIRY_TIME
            };
        } catch (error) {
            throw new Error('Authentication failed. Please check your credentials.');
        }
    }

    /**
     * Refreshes JWT token before expiration
     */
    public async refreshToken(): Promise<string> {
        const currentToken = this.tokenManager.getToken();
        if (!currentToken) {
            throw new Error('No token available for refresh');
        }

        try {
            const response = await axios.post(
                `${API_CONFIG.BASE_URL}/api/auth/refresh`,
                { token: currentToken }
            );

            const newToken = response.data.token;
            this.tokenManager.storeToken(newToken);
            return newToken;
        } catch (error) {
            this.logout();
            throw new Error('Token refresh failed');
        }
    }

    /**
     * Validates user permission for specific action
     */
    public async validatePermission(permission: Permission, resource?: string): Promise<boolean> {
        const user = this.sessionManager.getCurrentUser();
        if (!user) return false;

        if (this.sessionManager.isTokenExpired()) {
            try {
                await this.refreshToken();
            } catch {
                return false;
            }
        }

        const userPermissions = ROLE_PERMISSIONS[user.role];
        return userPermissions.includes(permission);
    }

    /**
     * Validates current user session
     */
    public async validateSession(): Promise<boolean> {
        const token = this.tokenManager.getToken();
        if (!token) return false;

        try {
            const decoded = jwtDecode<JWTToken>(token);
            if ((Date.now() / 1000) >= decoded.exp) {
                await this.refreshToken();
            }
            return true;
        } catch {
            this.logout();
            return false;
        }
    }

    /**
     * Logs out user and clears session
     */
    public logout(): void {
        this.tokenManager.removeToken();
        this.sessionManager.clearSession();
    }

    /**
     * Gets current authentication state
     */
    public getAuthState(): AuthState {
        const token = this.tokenManager.getToken();
        const user = this.sessionManager.getCurrentUser();
        
        return {
            isAuthenticated: !!token && !!user,
            user,
            token,
            tokenExpiry: this.sessionManager.isTokenExpired() ? null : TOKEN_EXPIRY_TIME
        };
    }
}

// Export singleton instance
export const authService = new AuthService();