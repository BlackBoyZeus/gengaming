/**
 * GameGen-X API Service
 * @version 1.0.0
 * 
 * Core API service module handling all HTTP/WebSocket communication with comprehensive
 * error handling, retry logic, circuit breakers, and performance monitoring.
 */

import axios, { AxiosInstance, AxiosError } from 'axios'; // v1.4.0
import axiosRetry from 'axios-retry'; // v3.5.0
import CircuitBreaker from 'opossum'; // v7.1.0

import {
  API_CONFIG,
  REQUEST_TIMEOUTS,
  RETRY_CONFIG,
  API_ENDPOINTS,
  WS_ENDPOINTS
} from '../config/api';

import {
  GenerationRequest,
  GenerationResponse,
  ControlRequest,
  ControlResponse,
  StatusResponse,
  SystemHealthStatus,
  GenerationStatus
} from '../types/api';

/**
 * Metrics collector for performance monitoring
 */
class MetricsCollector {
  private metrics: Map<string, number[]> = new Map();

  record(metric: string, value: number): void {
    if (!this.metrics.has(metric)) {
      this.metrics.set(metric, []);
    }
    this.metrics.get(metric)?.push(value);
  }

  getAverage(metric: string): number {
    const values = this.metrics.get(metric) || [];
    return values.length ? values.reduce((a, b) => a + b) / values.length : 0;
  }
}

/**
 * Core API service class implementing comprehensive error handling and monitoring
 */
export class APIService {
  private axios: AxiosInstance;
  private circuitBreaker: CircuitBreaker;
  private wsConnection: WebSocket | null = null;
  private metrics: MetricsCollector;
  private reconnectAttempts: number = 0;
  private readonly MAX_RECONNECT_ATTEMPTS = 3;

  constructor() {
    // Initialize axios instance with enhanced configuration
    this.axios = axios.create({
      baseURL: API_CONFIG.BASE_URL,
      timeout: REQUEST_TIMEOUTS.GENERATE,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // Configure retry strategy with exponential backoff
    axiosRetry(this.axios, {
      retries: RETRY_CONFIG.MAX_RETRIES,
      retryDelay: (retryCount) => {
        const delay = Math.min(
          RETRY_CONFIG.RETRY_DELAY * Math.pow(RETRY_CONFIG.BACKOFF_FACTOR, retryCount),
          REQUEST_TIMEOUTS.GENERATE
        );
        return delay + Math.random() * RETRY_CONFIG.JITTER_MAX;
      },
      retryCondition: (error: AxiosError) => {
        return axiosRetry.isNetworkOrIdempotentRequestError(error) ||
          RETRY_CONFIG.ERROR_CODES.includes(error.response?.status || 0);
      }
    });

    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker(async (request: () => Promise<any>) => {
      return await request();
    }, {
      timeout: REQUEST_TIMEOUTS.GENERATE,
      errorThresholdPercentage: 50,
      resetTimeout: 30000
    });

    // Initialize metrics collector
    this.metrics = new MetricsCollector();

    // Setup request interceptors
    this.setupInterceptors();
  }

  /**
   * Configure axios interceptors for monitoring and error handling
   */
  private setupInterceptors(): void {
    this.axios.interceptors.request.use((config) => {
      const requestStart = Date.now();
      config.metadata = { startTime: requestStart };
      return config;
    });

    this.axios.interceptors.response.use(
      (response) => {
        const duration = Date.now() - (response.config.metadata?.startTime || 0);
        this.metrics.record('requestDuration', duration);
        return response;
      },
      (error: AxiosError) => {
        const duration = Date.now() - (error.config?.metadata?.startTime || 0);
        this.metrics.record('errorDuration', duration);
        throw this.enhanceError(error);
      }
    );
  }

  /**
   * Enhance error with additional context and classification
   */
  private enhanceError(error: AxiosError): Error {
    const enhancedError = new Error(error.message);
    enhancedError.name = this.classifyError(error);
    return enhancedError;
  }

  /**
   * Classify error type for appropriate handling
   */
  private classifyError(error: AxiosError): string {
    if (!error.response) return 'NetworkError';
    switch (error.response.status) {
      case 429: return 'RateLimitError';
      case 503: return 'ServiceUnavailableError';
      default: return 'APIError';
    }
  }

  /**
   * Generate video with comprehensive error handling and monitoring
   */
  public async generateVideo(request: GenerationRequest): Promise<GenerationResponse> {
    const startTime = Date.now();

    try {
      const response = await this.circuitBreaker.fire(async () => {
        return await this.axios.post<GenerationResponse>(
          API_ENDPOINTS.GENERATE,
          request,
          { timeout: REQUEST_TIMEOUTS.GENERATE }
        );
      });

      const duration = Date.now() - startTime;
      this.metrics.record('generateDuration', duration);

      return response.data;
    } catch (error) {
      this.handleGenerationError(error as Error);
      throw error;
    }
  }

  /**
   * Send control command with real-time monitoring
   */
  public async sendControl(request: ControlRequest): Promise<ControlResponse> {
    const startTime = Date.now();

    try {
      const response = await this.axios.post<ControlResponse>(
        API_ENDPOINTS.CONTROL,
        request,
        { timeout: REQUEST_TIMEOUTS.CONTROL }
      );

      const duration = Date.now() - startTime;
      this.metrics.record('controlDuration', duration);

      return response.data;
    } catch (error) {
      this.handleControlError(error as Error);
      throw error;
    }
  }

  /**
   * Get system status with health checks
   */
  public async getStatus(): Promise<StatusResponse> {
    try {
      const response = await this.axios.get<StatusResponse>(
        API_ENDPOINTS.STATUS,
        { timeout: REQUEST_TIMEOUTS.STATUS }
      );
      return response.data;
    } catch (error) {
      this.handleStatusError(error as Error);
      throw error;
    }
  }

  /**
   * Setup WebSocket connection with reconnection logic
   */
  public async setupWebSocket(): Promise<void> {
    if (this.wsConnection) {
      this.wsConnection.close();
    }

    this.wsConnection = new WebSocket(`${API_CONFIG.WS_URL}${WS_ENDPOINTS.STREAM}`);
    
    this.wsConnection.onopen = () => {
      this.reconnectAttempts = 0;
      this.startHeartbeat();
    };

    this.wsConnection.onclose = () => {
      this.handleWebSocketClose();
    };

    this.wsConnection.onerror = (error) => {
      this.handleWebSocketError(error);
    };

    this.wsConnection.onmessage = (event) => {
      this.handleWebSocketMessage(event);
    };
  }

  /**
   * Start WebSocket heartbeat monitoring
   */
  private startHeartbeat(): void {
    const HEARTBEAT_INTERVAL = 30000;
    setInterval(() => {
      if (this.wsConnection?.readyState === WebSocket.OPEN) {
        this.wsConnection.send(JSON.stringify({ type: 'ping' }));
      }
    }, HEARTBEAT_INTERVAL);
  }

  /**
   * Handle WebSocket connection closure
   */
  private handleWebSocketClose(): void {
    if (this.reconnectAttempts < this.MAX_RECONNECT_ATTEMPTS) {
      this.reconnectAttempts++;
      setTimeout(() => this.setupWebSocket(), 5000 * this.reconnectAttempts);
    }
  }

  /**
   * Handle WebSocket errors
   */
  private handleWebSocketError(error: Event): void {
    this.metrics.record('wsErrors', 1);
    console.error('WebSocket error:', error);
  }

  /**
   * Process incoming WebSocket messages
   */
  private handleWebSocketMessage(event: MessageEvent): void {
    try {
      const message = JSON.parse(event.data);
      // Handle different message types (frame updates, control responses, etc.)
    } catch (error) {
      console.error('Error processing WebSocket message:', error);
    }
  }

  /**
   * Handle video generation errors
   */
  private handleGenerationError(error: Error): void {
    this.metrics.record('generationErrors', 1);
    // Additional error handling logic
  }

  /**
   * Handle control command errors
   */
  private handleControlError(error: Error): void {
    this.metrics.record('controlErrors', 1);
    // Additional error handling logic
  }

  /**
   * Handle status check errors
   */
  private handleStatusError(error: Error): void {
    this.metrics.record('statusErrors', 1);
    // Additional error handling logic
  }
}

export default new APIService();