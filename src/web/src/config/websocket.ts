/**
 * WebSocket Configuration for GameGen-X
 * @version 1.0.0
 * 
 * Provides secure WebSocket configuration for real-time video streaming
 * and control interactions with comprehensive validation and monitoring.
 */

import { ControlType } from '../types/api';
import { WEBSOCKET_CONFIG } from './constants';

// Base WebSocket URL with environment-based configuration
export const WS_BASE_URL = process.env.VITE_WS_BASE_URL || 'ws://localhost:8000';

/**
 * WebSocket message types with enhanced validation support
 */
export const WS_MESSAGE_TYPES = {
    FRAME: 'frame',
    CONTROL: 'control',
    STATUS: 'status',
    ERROR: 'error',
    PING: 'ping',
    PONG: 'pong',
    BINARY_FRAME: 'binary_frame',
    METRICS: 'metrics'
} as const;

/**
 * Extended WebSocket close event codes with detailed handling
 */
export const WS_CLOSE_CODES = {
    NORMAL: 1000,
    GOING_AWAY: 1001,
    PROTOCOL_ERROR: 1002,
    INVALID_DATA: 1003,
    POLICY_VIOLATION: 1008,
    MESSAGE_TOO_BIG: 1009,
    INTERNAL_ERROR: 1011,
    RATE_LIMITED: 4000,
    VALIDATION_ERROR: 4001,
    AUTHENTICATION_ERROR: 4002
} as const;

/**
 * WebSocket event names with performance monitoring
 */
export const WS_EVENTS = {
    OPEN: 'open',
    MESSAGE: 'message',
    ERROR: 'error',
    CLOSE: 'close',
    LATENCY: 'latency',
    RECONNECTING: 'reconnecting',
    DEGRADED: 'degraded'
} as const;

/**
 * Interface for WebSocket connection options
 */
interface ConnectionOptions {
    secure?: boolean;
    headers?: Record<string, string>;
    protocols?: string[];
    timeout?: number;
    maxMessageSize?: number;
}

/**
 * Interface for message validation options
 */
interface ValidationOptions {
    maxSize?: number;
    requiredFields?: string[];
    allowedTypes?: string[];
    customValidators?: ((message: unknown) => boolean)[];
}

/**
 * Interface for validation result
 */
interface ValidationResult {
    valid: boolean;
    errors: string[];
    warnings: string[];
}

/**
 * Main WebSocket configuration object with security and performance settings
 */
export const WS_CONFIG = {
    BASE_URL: WS_BASE_URL,
    RECONNECT_INTERVAL: WEBSOCKET_CONFIG.RECONNECT_INTERVAL,
    MAX_RETRIES: WEBSOCKET_CONFIG.MAX_RETRIES,
    PING_INTERVAL: WEBSOCKET_CONFIG.PING_INTERVAL,
    MAX_MESSAGE_SIZE: 1024 * 1024 * 5, // 5MB max message size
    PERFORMANCE_THRESHOLDS: {
        MAX_LATENCY: 100, // ms
        MIN_FPS: 24,
        DEGRADED_LATENCY: 200, // ms
        CRITICAL_LATENCY: 500 // ms
    },
    SECURITY: {
        REQUIRE_SECURE: process.env.NODE_ENV === 'production',
        VALIDATE_ORIGIN: true,
        RATE_LIMIT: {
            MAX_MESSAGES_PER_SECOND: 60,
            BURST_SIZE: 100
        }
    }
} as const;

/**
 * Creates a secure WebSocket URL with enhanced validation and sanitization
 * @param path - WebSocket endpoint path
 * @param params - Query parameters
 * @param options - Connection options
 * @returns Sanitized and validated WebSocket URL
 */
export function createWebSocketURL(
    path: string,
    params: Record<string, string> = {},
    options: ConnectionOptions = {}
): string {
    // Validate and sanitize path
    const sanitizedPath = path.replace(/[^\w\-/]/g, '');
    
    // Build base URL with protocol
    const protocol = options.secure || WS_CONFIG.SECURITY.REQUIRE_SECURE ? 'wss://' : 'ws://';
    const baseUrl = WS_BASE_URL.replace(/^(ws|wss):\/\//, '');
    
    // Encode and validate query parameters
    const queryParams = new URLSearchParams({
        ...params,
        v: 'v1',
        t: Date.now().toString()
    });

    // Add performance monitoring parameters
    if (options.timeout) {
        queryParams.append('timeout', options.timeout.toString());
    }

    // Construct final URL
    const url = `${protocol}${baseUrl}${sanitizedPath}?${queryParams.toString()}`;

    // Validate URL format
    if (!url.match(/^(ws|wss):\/\/[\w\-\.]+(:\d+)?(\/[\w\-\/]*)?(\?.*)?$/)) {
        throw new Error('Invalid WebSocket URL format');
    }

    return url;
}

/**
 * Enhanced message validation with size limits and type checking
 * @param message - Message to validate
 * @param options - Validation options
 * @returns Validation result with detailed information
 */
export function validateMessage(
    message: unknown,
    options: ValidationOptions = {}
): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Size validation
    const messageSize = new Blob([JSON.stringify(message)]).size;
    if (messageSize > (options.maxSize || WS_CONFIG.MAX_MESSAGE_SIZE)) {
        errors.push(`Message size ${messageSize} exceeds maximum allowed size`);
    }

    // Type validation
    if (typeof message === 'object' && message !== null) {
        const msg = message as Record<string, unknown>;
        
        // Required fields check
        if (options.requiredFields) {
            for (const field of options.requiredFields) {
                if (!(field in msg)) {
                    errors.push(`Missing required field: ${field}`);
                }
            }
        }

        // Message type validation
        if (options.allowedTypes && 'type' in msg) {
            if (!options.allowedTypes.includes(msg.type as string)) {
                errors.push(`Invalid message type: ${msg.type}`);
            }
        }

        // Custom validation
        if (options.customValidators) {
            for (const validator of options.customValidators) {
                if (!validator(message)) {
                    errors.push('Message failed custom validation');
                }
            }
        }
    } else {
        errors.push('Message must be an object');
    }

    return {
        valid: errors.length === 0,
        errors,
        warnings
    };
}

// Type exports for enhanced type safety
export type WebSocketMessageTypes = typeof WS_MESSAGE_TYPES;
export type WebSocketCloseCode = typeof WS_CLOSE_CODES;
export type WebSocketEvent = typeof WS_EVENTS;
export type WebSocketConfig = typeof WS_CONFIG;