/**
 * WebSocket Service Implementation for GameGen-X
 * @version 1.0.0
 * 
 * Provides secure WebSocket management for real-time video streaming and control
 * with comprehensive performance monitoring and error handling.
 */

import { WS_CONFIG, WS_MESSAGE_TYPES, validateMessage, createWebSocketURL } from '../config/websocket';
import { ControlType } from '../types/api';
import pako from 'pako'; // v2.1.0
import CryptoJS from 'crypto-js'; // v4.1.1
import { PerformanceMonitor } from 'performance-monitor'; // v1.0.0

/**
 * Interface for WebSocket message handlers
 */
interface MessageHandler {
    (data: any): Promise<void>;
}

/**
 * Interface for WebSocket options
 */
interface WebSocketOptions {
    compression?: boolean;
    encryption?: boolean;
    performanceMonitoring?: boolean;
    validateMessages?: boolean;
}

/**
 * Enhanced WebSocket manager with security and performance monitoring
 */
export class WebSocketManager {
    private ws: WebSocket | null = null;
    private retryCount = 0;
    private isConnected = false;
    private performanceMonitor: PerformanceMonitor;
    private lastPingTime = 0;
    private pingInterval: NodeJS.Timer | null = null;
    private messageHandlers: Map<string, MessageHandler> = new Map();
    private options: WebSocketOptions;
    private readonly generationId: string;
    private readonly token: string;

    constructor(generationId: string, token: string, options: WebSocketOptions = {}) {
        this.generationId = generationId;
        this.token = token;
        this.options = {
            compression: true,
            encryption: true,
            performanceMonitoring: true,
            validateMessages: true,
            ...options
        };

        this.performanceMonitor = new PerformanceMonitor({
            maxLatency: WS_CONFIG.PERFORMANCE_THRESHOLDS.MAX_LATENCY,
            minFps: WS_CONFIG.PERFORMANCE_THRESHOLDS.MIN_FPS
        });

        // Initialize message handlers
        this.initializeMessageHandlers();
    }

    /**
     * Establishes secure WebSocket connection with monitoring
     */
    public async connect(): Promise<void> {
        try {
            const url = createWebSocketURL(`/ws/${this.generationId}`, {
                token: this.token
            });

            this.ws = new WebSocket(url);
            this.setupEventHandlers();

            if (this.options.performanceMonitoring) {
                this.performanceMonitor.start();
            }

            // Initialize ping interval for connection monitoring
            this.initializePingInterval();

        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.handleConnectionError(error);
        }
    }

    /**
     * Safely disconnects WebSocket and cleans up resources
     */
    public disconnect(): void {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }

        if (this.options.performanceMonitoring) {
            this.performanceMonitor.stop();
        }

        if (this.ws) {
            this.ws.close(1000, 'Client disconnecting');
            this.ws = null;
        }

        this.isConnected = false;
        this.messageHandlers.clear();
    }

    /**
     * Sends control signal with validation and monitoring
     */
    public async sendControl(type: ControlType, data: Record<string, unknown>): Promise<void> {
        if (!this.isConnected || !this.ws) {
            throw new Error('WebSocket not connected');
        }

        const message = {
            type: WS_MESSAGE_TYPES.CONTROL,
            controlType: type,
            data,
            timestamp: Date.now()
        };

        if (this.options.validateMessages) {
            const validation = validateMessage(message, {
                maxSize: WS_CONFIG.MAX_MESSAGE_SIZE,
                requiredFields: ['type', 'controlType', 'data']
            });

            if (!validation.valid) {
                throw new Error(`Invalid message: ${validation.errors.join(', ')}`);
            }
        }

        await this.sendMessage(message);
    }

    /**
     * Processes incoming messages with validation and monitoring
     */
    private async handleMessage(event: MessageEvent): Promise<void> {
        try {
            const startTime = performance.now();
            let data = event.data;

            // Decompress if needed
            if (this.options.compression && typeof data === 'string') {
                data = pako.inflate(data, { to: 'string' });
            }

            // Decrypt if needed
            if (this.options.encryption && typeof data === 'string') {
                const bytes = CryptoJS.AES.decrypt(data, this.token);
                data = bytes.toString(CryptoJS.enc.Utf8);
            }

            const message = JSON.parse(data);

            if (this.options.validateMessages) {
                const validation = validateMessage(message, {
                    maxSize: WS_CONFIG.MAX_MESSAGE_SIZE,
                    requiredFields: ['type']
                });

                if (!validation.valid) {
                    throw new Error(`Invalid message: ${validation.errors.join(', ')}`);
                }
            }

            const handler = this.messageHandlers.get(message.type);
            if (handler) {
                await handler(message);
            }

            // Update performance metrics
            if (this.options.performanceMonitoring) {
                const processingTime = performance.now() - startTime;
                this.performanceMonitor.recordMetric('messageProcessing', processingTime);
            }

        } catch (error) {
            console.error('Message handling error:', error);
            this.handleMessageError(error);
        }
    }

    /**
     * Sets up WebSocket event handlers with error recovery
     */
    private setupEventHandlers(): void {
        if (!this.ws) return;

        this.ws.onopen = () => {
            this.isConnected = true;
            this.retryCount = 0;
            console.log('WebSocket connected');
        };

        this.ws.onclose = (event) => {
            this.isConnected = false;
            this.handleConnectionClose(event);
        };

        this.ws.onerror = (error) => {
            this.handleConnectionError(error);
        };

        this.ws.onmessage = (event) => {
            this.handleMessage(event);
        };
    }

    /**
     * Initializes message handlers for different message types
     */
    private initializeMessageHandlers(): void {
        this.messageHandlers.set(WS_MESSAGE_TYPES.FRAME, async (data) => {
            // Handle video frame data
            if (this.options.performanceMonitoring) {
                this.performanceMonitor.recordFrame();
            }
        });

        this.messageHandlers.set(WS_MESSAGE_TYPES.STATUS, async (data) => {
            // Handle status updates
            if (this.options.performanceMonitoring) {
                this.performanceMonitor.recordMetric('status', data.metrics);
            }
        });

        this.messageHandlers.set(WS_MESSAGE_TYPES.ERROR, async (data) => {
            console.error('WebSocket error:', data.error);
        });

        this.messageHandlers.set(WS_MESSAGE_TYPES.PING, async () => {
            await this.sendMessage({ type: WS_MESSAGE_TYPES.PONG });
        });
    }

    /**
     * Sends message with compression and encryption
     */
    private async sendMessage(message: unknown): Promise<void> {
        if (!this.ws || !this.isConnected) {
            throw new Error('WebSocket not connected');
        }

        let data = JSON.stringify(message);

        if (this.options.encryption) {
            data = CryptoJS.AES.encrypt(data, this.token).toString();
        }

        if (this.options.compression) {
            data = pako.deflate(data, { to: 'string' });
        }

        this.ws.send(data);
    }

    /**
     * Initializes ping interval for connection monitoring
     */
    private initializePingInterval(): void {
        this.pingInterval = setInterval(async () => {
            if (this.isConnected) {
                this.lastPingTime = Date.now();
                await this.sendMessage({ type: WS_MESSAGE_TYPES.PING });
            }
        }, WS_CONFIG.PING_INTERVAL);
    }

    /**
     * Handles connection close with retry logic
     */
    private handleConnectionClose(event: CloseEvent): void {
        if (event.code !== 1000 && this.retryCount < WS_CONFIG.MAX_RETRIES) {
            setTimeout(() => {
                this.retryCount++;
                this.connect();
            }, WS_CONFIG.RECONNECT_INTERVAL);
        }
    }

    /**
     * Handles connection errors with monitoring
     */
    private handleConnectionError(error: any): void {
        console.error('WebSocket error:', error);
        if (this.options.performanceMonitoring) {
            this.performanceMonitor.recordError('connection');
        }
    }

    /**
     * Handles message processing errors
     */
    private handleMessageError(error: any): void {
        console.error('Message error:', error);
        if (this.options.performanceMonitoring) {
            this.performanceMonitor.recordError('message');
        }
    }
}

/**
 * Creates and configures a new secure WebSocket connection
 */
export function createSecureWebSocketConnection(
    generationId: string,
    token: string,
    options: WebSocketOptions = {}
): WebSocketManager {
    return new WebSocketManager(generationId, token, options);
}