/**
 * GameGen-X Web Interface Constants
 * Version: 1.0.0
 * 
 * Core system constants and configuration values enforcing technical specifications
 * across the GameGen-X web application.
 */

// Environment configuration
export const ENVIRONMENT = process.env.NODE_ENV || 'development';

/**
 * System performance and resource limits enforcing technical specifications
 * from section 2.4.4 of technical requirements
 */
export const SYSTEM_LIMITS = {
  MAX_GENERATION_LATENCY: 100, // Maximum allowed generation latency in ms
  MIN_FRAME_RATE: 24, // Minimum required frame rate
  MAX_CONTROL_LATENCY: 50, // Maximum allowed control response time in ms
  MAX_CONCURRENT_USERS: 100, // Maximum concurrent users supported
  MAX_MEMORY_USAGE: '512GB', // Maximum system memory allocation
  MAX_GPU_MEMORY: '80GB' // Maximum GPU memory allocation
} as const;

/**
 * Video generation and playback configuration aligned with system capabilities
 * from section 1.3.1 of technical requirements
 */
export const VIDEO_SETTINGS = {
  DEFAULT_RESOLUTION: {
    width: 1280,
    height: 720
  },
  DEFAULT_FRAME_COUNT: 102,
  DEFAULT_FPS: 24,
  SUPPORTED_RESOLUTIONS: [
    { width: 320, height: 256, maxFrames: 200 },
    { width: 848, height: 480, maxFrames: 150 },
    { width: 1280, height: 720, maxFrames: 102 }
  ],
  VIDEO_FORMATS: ['mp4', 'webm'] as const,
  CODEC_SETTINGS: {
    video: 'h264',
    audio: null
  }
} as const;

/**
 * WebSocket connection parameters for real-time communication
 * from section 2.1 High-Level Architecture
 */
export const WEBSOCKET_CONFIG = {
  RECONNECT_INTERVAL: 5000, // Reconnection attempt interval in ms
  MAX_RETRIES: 3, // Maximum reconnection attempts
  PING_INTERVAL: 30000, // WebSocket ping interval in ms
  PONG_TIMEOUT: 5000, // Maximum time to wait for pong response
  CLOSE_CODE: 1000, // Normal closure code
  CLOSE_TIMEOUT: 3000, // Connection close timeout in ms
  MESSAGE_TYPES: {
    FRAME: 'frame',
    CONTROL: 'control',
    ERROR: 'error',
    STATUS: 'status'
  }
} as const;

/**
 * Control and interaction parameters for responsive user experience
 * Ensures compliance with control response requirements
 */
export const CONTROL_SETTINGS = {
  KEYBOARD_UPDATE_RATE: 60, // Keyboard input polling rate in Hz
  ENVIRONMENT_UPDATE_RATE: 30, // Environment update rate in Hz
  MIN_CONTROL_INTERVAL: 50, // Minimum interval between control updates in ms
  MAX_QUEUE_SIZE: 100, // Maximum control queue size
  DEBOUNCE_DELAY: 16, // Input debounce delay in ms (~60fps)
  INPUT_BUFFER_SIZE: 8, // Size of input buffer for smooth control
  CONTROL_MODES: {
    KEYBOARD: 'keyboard',
    GAMEPAD: 'gamepad',
    TOUCH: 'touch'
  }
} as const;

/**
 * User interface configuration for consistent visual experience
 * Defines UI behavior and appearance constants
 */
export const UI_CONSTANTS = {
  ANIMATION_DURATION: 200, // Default animation duration in ms
  TOAST_DURATION: 3000, // Notification display duration in ms
  MAX_PROMPT_LENGTH: 300, // Maximum text prompt length
  MIN_PROMPT_LENGTH: 10, // Minimum text prompt length
  ERROR_DISPLAY_DURATION: 5000, // Error message display duration in ms
  LOADING_STATES: {
    INITIAL: 'initial',
    LOADING: 'loading',
    SUCCESS: 'success',
    ERROR: 'error'
  },
  THEME: {
    DARK: 'dark',
    LIGHT: 'light',
    SYSTEM: 'system'
  }
} as const;

// Type exports for type-safe usage
export type SystemLimits = typeof SYSTEM_LIMITS;
export type VideoSettings = typeof VIDEO_SETTINGS;
export type WebSocketConfig = typeof WEBSOCKET_CONFIG;
export type ControlSettings = typeof CONTROL_SETTINGS;
export type UIConstants = typeof UI_CONSTANTS;

// Ensure immutability of all constants
Object.freeze(SYSTEM_LIMITS);
Object.freeze(VIDEO_SETTINGS);
Object.freeze(WEBSOCKET_CONFIG);
Object.freeze(CONTROL_SETTINGS);
Object.freeze(UI_CONSTANTS);