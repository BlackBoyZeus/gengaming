/**
 * GameGen-X API Configuration
 * @version 1.0.0
 * 
 * Defines API endpoints, timeouts, and retry configurations for the GameGen-X web interface
 * Implements specifications from Technical Specifications sections 2.3.1, 2.3.2, and 5.3.5
 */

import { ControlType } from '../types/api';
import { SYSTEM_LIMITS } from './constants';

// API version and base URL configuration
export const API_VERSION = 'v1';
export const API_BASE_URL = process.env.VITE_API_BASE_URL || 'http://localhost:8000';
export const WS_BASE_URL = process.env.VITE_WS_BASE_URL || 'ws://localhost:8000';

/**
 * Core API configuration object
 */
export const API_CONFIG = {
  BASE_URL: API_BASE_URL,
  WS_URL: WS_BASE_URL,
  VERSION: API_VERSION,
} as const;

/**
 * REST API endpoint configurations
 * Aligned with FastAPI backend routes
 */
export const API_ENDPOINTS = {
  GENERATE: `/api/${API_VERSION}/generate`,
  CONTROL: `/api/${API_VERSION}/control`,
  STATUS: `/api/${API_VERSION}/status`,
} as const;

/**
 * WebSocket endpoint configurations
 * For real-time video streaming and control
 */
export const WS_ENDPOINTS = {
  STREAM: '/ws/stream',
} as const;

/**
 * Request timeout configurations (in milliseconds)
 * Aligned with performance requirements from technical specifications
 */
export const REQUEST_TIMEOUTS = {
  // Generation timeout aligned with MAX_GENERATION_LATENCY * safety factor
  GENERATE: SYSTEM_LIMITS.MAX_GENERATION_LATENCY * 1000,
  // Control timeout for real-time responsiveness
  CONTROL: 5000,
  // Status endpoint timeout
  STATUS: 3000,
} as const;

/**
 * Enhanced retry configuration for API reliability
 * Implements exponential backoff with jitter
 */
export const RETRY_CONFIG = {
  // Maximum number of retry attempts
  MAX_RETRIES: 3,
  // Base delay between retries in milliseconds
  RETRY_DELAY: 1000,
  // Exponential backoff factor
  BACKOFF_FACTOR: 2,
  // Maximum jitter in milliseconds for retry randomization
  JITTER_MAX: 100,
  // Timeout multiplier for subsequent retries
  TIMEOUT_MULTIPLIER: 1.5,
  // HTTP status codes that trigger retry attempts
  ERROR_CODES: [
    408, // Request Timeout
    429, // Too Many Requests
    500, // Internal Server Error
    502, // Bad Gateway
    503, // Service Unavailable
    504, // Gateway Timeout
  ],
} as const;

/**
 * Type definitions for API configuration exports
 */
export type ApiConfig = typeof API_CONFIG;
export type ApiEndpoints = typeof API_ENDPOINTS;
export type WsEndpoints = typeof WS_ENDPOINTS;
export type RequestTimeouts = typeof REQUEST_TIMEOUTS;
export type RetryConfig = typeof RETRY_CONFIG;

// Ensure immutability of all configuration objects
Object.freeze(API_CONFIG);
Object.freeze(API_ENDPOINTS);
Object.freeze(WS_ENDPOINTS);
Object.freeze(REQUEST_TIMEOUTS);
Object.freeze(RETRY_CONFIG);