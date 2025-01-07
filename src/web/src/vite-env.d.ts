/// <reference types="vite/client" /> /* vite ^4.0.0 */

/**
 * Environment variable type definitions for the GameGen-X web application.
 * Extends Vite's base ImportMetaEnv interface with application-specific configurations.
 */
interface ImportMetaEnv {
  /**
   * Base URL for the GameGen-X API endpoints
   * @example 'https://api.gamegen-x.com'
   */
  readonly VITE_API_URL: string;

  /**
   * WebSocket URL for real-time video streaming and control
   * @example 'wss://ws.gamegen-x.com'
   */
  readonly VITE_WS_URL: string;

  /**
   * Current environment name
   * Supports development, production, and test environments
   */
  readonly VITE_ENV: 'development' | 'production' | 'test';
}

/**
 * Augments Vite's ImportMeta interface to include GameGen-X environment variables
 * Ensures type safety when accessing import.meta.env throughout the application
 */
interface ImportMeta {
  readonly env: ImportMetaEnv;
}