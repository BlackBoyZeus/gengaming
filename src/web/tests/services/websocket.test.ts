import { describe, test, expect, jest, beforeEach, afterEach } from '@jest/globals';
import WS from 'jest-websocket-mock';
import now from 'performance-now';
import { 
    createWebSocketConnection, 
    sendControlSignal, 
    WebSocketManager 
} from '../../src/services/websocket';
import { 
    WS_CONFIG, 
    WS_MESSAGE_TYPES,
    createWebSocketURL,
    validateMessage
} from '../../src/config/websocket';
import { ControlType, SystemHealthStatus } from '../../src/types/api';

// Test configuration constants
const TEST_CONFIG = {
    GENERATION_ID: 'test-generation-123',
    AUTH_TOKEN: 'test-auth-token-456',
    WS_TEST_PORT: 8080,
    LATENCY_THRESHOLD: 50, // ms
    FRAME_RATE_TARGET: 24,
    TEST_TIMEOUT: 5000, // ms
    COMPRESSION_SIZE: 1024 // bytes
};

let wsServer: WS;
let wsManager: WebSocketManager;
let performanceMetrics: {
    latencies: number[];
    frameRates: number[];
    errors: string[];
};

beforeEach(async () => {
    // Initialize test WebSocket server
    wsServer = new WS(
        `ws://localhost:${TEST_CONFIG.WS_TEST_PORT}/ws/${TEST_CONFIG.GENERATION_ID}`,
        { jsonProtocol: true }
    );

    // Initialize performance metrics
    performanceMetrics = {
        latencies: [],
        frameRates: [],
        errors: []
    };

    // Create WebSocket manager with test configuration
    wsManager = new WebSocketManager(
        TEST_CONFIG.GENERATION_ID,
        TEST_CONFIG.AUTH_TOKEN,
        {
            compression: true,
            encryption: true,
            performanceMonitoring: true,
            validateMessages: true
        }
    );

    await wsManager.connect();
});

afterEach(() => {
    // Clean up resources
    WS.clean();
    wsManager.disconnect();
    jest.clearAllMocks();
});

describe('WebSocket Connection Tests', () => {
    test('should establish secure connection with valid token', async () => {
        const connection = await createWebSocketConnection(
            TEST_CONFIG.GENERATION_ID,
            TEST_CONFIG.AUTH_TOKEN
        );
        
        expect(connection).toBeDefined();
        expect(wsServer.connected).toBe(true);
        
        // Verify secure connection parameters
        const url = createWebSocketURL(`/ws/${TEST_CONFIG.GENERATION_ID}`, {
            token: TEST_CONFIG.AUTH_TOKEN
        });
        expect(url.startsWith('wss://')).toBe(true);
    });

    test('should reject connection with invalid token', async () => {
        await expect(
            createWebSocketConnection(TEST_CONFIG.GENERATION_ID, 'invalid-token')
        ).rejects.toThrow('Authentication failed');
    });

    test('should implement reconnection strategy with backoff', async () => {
        const reconnectSpy = jest.spyOn(wsManager as any, 'connect');
        wsServer.close();
        
        // Wait for reconnection attempts
        await new Promise(resolve => setTimeout(resolve, WS_CONFIG.RECONNECT_INTERVAL * 2));
        
        expect(reconnectSpy).toHaveBeenCalledTimes(2);
        expect(wsManager['retryCount']).toBeLessThanOrEqual(WS_CONFIG.MAX_RETRIES);
    });
});

describe('Control Signal Tests', () => {
    test('should transmit control signals within latency threshold', async () => {
        const controlData = {
            type: ControlType.KEYBOARD,
            data: { key: 'W', pressed: true }
        };

        const startTime = now();
        await wsManager.sendControl(controlData.type, controlData.data);
        const endTime = now();
        
        const latency = endTime - startTime;
        performanceMetrics.latencies.push(latency);
        
        expect(latency).toBeLessThan(TEST_CONFIG.LATENCY_THRESHOLD);
        expect(wsServer).toHaveReceivedMessage(
            expect.objectContaining({
                type: WS_MESSAGE_TYPES.CONTROL,
                controlType: controlData.type,
                data: controlData.data
            })
        );
    });

    test('should handle message compression effectively', async () => {
        const largeControlData = {
            type: ControlType.ENVIRONMENT,
            data: { 
                state: 'A'.repeat(TEST_CONFIG.COMPRESSION_SIZE)
            }
        };

        const messageSpy = jest.spyOn(wsManager['ws'] as WebSocket, 'send');
        await wsManager.sendControl(largeControlData.type, largeControlData.data);
        
        const sentMessage = messageSpy.mock.calls[0][0] as string;
        expect(sentMessage.length).toBeLessThan(TEST_CONFIG.COMPRESSION_SIZE);
    });
});

describe('Message Processing Tests', () => {
    test('should handle video frame messages at target FPS', async () => {
        const frameCount = TEST_CONFIG.FRAME_RATE_TARGET;
        const startTime = now();
        
        for (let i = 0; i < frameCount; i++) {
            wsServer.send({
                type: WS_MESSAGE_TYPES.FRAME,
                data: { frameId: i, content: 'frame-data' }
            });
        }
        
        const endTime = now();
        const achievedFPS = frameCount / ((endTime - startTime) / 1000);
        performanceMetrics.frameRates.push(achievedFPS);
        
        expect(achievedFPS).toBeGreaterThanOrEqual(TEST_CONFIG.FRAME_RATE_TARGET);
    });

    test('should validate message integrity and format', async () => {
        const invalidMessage = {
            type: 'invalid_type',
            data: {}
        };

        const validation = validateMessage(invalidMessage, {
            maxSize: WS_CONFIG.MAX_MESSAGE_SIZE,
            requiredFields: ['type', 'data'],
            allowedTypes: Object.values(WS_MESSAGE_TYPES)
        });

        expect(validation.valid).toBe(false);
        expect(validation.errors).toContain('Invalid message type: invalid_type');
    });

    test('should handle error messages appropriately', async () => {
        const errorSpy = jest.spyOn(console, 'error');
        
        wsServer.send({
            type: WS_MESSAGE_TYPES.ERROR,
            error: 'Test error message'
        });

        expect(errorSpy).toHaveBeenCalledWith(
            'WebSocket error:',
            'Test error message'
        );
    });
});

describe('Performance Monitoring Tests', () => {
    test('should monitor and report performance metrics', async () => {
        const metrics = {
            latency: Math.max(...performanceMetrics.latencies),
            frameRate: Math.min(...performanceMetrics.frameRates),
            errors: performanceMetrics.errors.length
        };

        expect(metrics.latency).toBeLessThan(TEST_CONFIG.LATENCY_THRESHOLD);
        expect(metrics.frameRate).toBeGreaterThanOrEqual(TEST_CONFIG.FRAME_RATE_TARGET);
        expect(metrics.errors).toBe(0);
    });

    test('should detect and report degraded performance', async () => {
        const statusSpy = jest.spyOn(wsManager as any, 'handleMessage');
        
        wsServer.send({
            type: WS_MESSAGE_TYPES.STATUS,
            status: SystemHealthStatus.DEGRADED,
            metrics: {
                latency: TEST_CONFIG.LATENCY_THRESHOLD * 2,
                frameRate: TEST_CONFIG.FRAME_RATE_TARGET / 2
            }
        });

        expect(statusSpy).toHaveBeenCalled();
        expect(performanceMetrics.errors).toContain('Performance degraded');
    });
});