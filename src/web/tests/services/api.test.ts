import { APIService } from '../../src/services/api';
import { GenerationRequest, ControlRequest, GenerationStatus, SystemHealthStatus } from '../../src/types/api';
import { SYSTEM_LIMITS, WEBSOCKET_CONFIG } from '../../src/config/constants';
import { API_CONFIG, REQUEST_TIMEOUTS } from '../../src/config/api';
import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';
import { WebSocket, Server } from 'mock-socket';
import { advanceTimersByTime, useFakeTimers } from '@jest/fake-timers';

describe('APIService', () => {
  let apiService: APIService;
  let mockAxios: MockAdapter;
  let mockWebSocketServer: Server;
  let originalWebSocket: typeof WebSocket;

  beforeAll(() => {
    originalWebSocket = global.WebSocket;
    global.WebSocket = WebSocket as any;
  });

  afterAll(() => {
    global.WebSocket = originalWebSocket;
  });

  beforeEach(() => {
    jest.useFakeTimers();
    mockAxios = new MockAdapter(axios);
    apiService = new APIService();
    mockWebSocketServer = new Server(`${API_CONFIG.WS_URL}${WEBSOCKET_CONFIG.MESSAGE_TYPES.FRAME}`);
  });

  afterEach(() => {
    mockAxios.restore();
    mockWebSocketServer.stop();
    jest.clearAllTimers();
  });

  describe('generateVideo', () => {
    const validRequest: GenerationRequest = {
      prompt: 'Test video generation',
      parameters: {
        resolution: { width: 1280, height: 720 },
        frames: 102,
        fps: 24,
        perspective: 'THIRD_PERSON'
      },
      timestamp: new Date().toISOString()
    };

    it('should successfully generate video within latency requirements', async () => {
      const mockResponse = {
        id: '123',
        status: GenerationStatus.COMPLETED,
        metrics: { generation_time_ms: 50 }
      };

      mockAxios.onPost('/api/v1/generate').reply(200, mockResponse);

      const startTime = Date.now();
      const response = await apiService.generateVideo(validRequest);
      const duration = Date.now() - startTime;

      expect(response).toEqual(mockResponse);
      expect(duration).toBeLessThan(SYSTEM_LIMITS.MAX_GENERATION_LATENCY);
    });

    it('should handle rate limiting with exponential backoff', async () => {
      mockAxios.onPost('/api/v1/generate')
        .replyOnce(429)
        .replyOnce(429)
        .replyOnce(200, { id: '123', status: GenerationStatus.COMPLETED });

      const response = await apiService.generateVideo(validRequest);
      expect(response.id).toBe('123');
    });

    it('should activate circuit breaker after consecutive failures', async () => {
      mockAxios.onPost('/api/v1/generate').reply(500);

      for (let i = 0; i < 5; i++) {
        try {
          await apiService.generateVideo(validRequest);
        } catch (error) {
          expect(error.name).toBe('ServiceUnavailableError');
        }
      }

      // Circuit breaker should be open
      await expect(apiService.generateVideo(validRequest))
        .rejects.toThrow('Circuit breaker is open');
    });
  });

  describe('sendControl', () => {
    const validControl: ControlRequest = {
      type: 'KEYBOARD',
      data: { key: 'W' },
      generation_id: '123',
      video_id: '456',
      timestamp: new Date().toISOString()
    };

    it('should send control commands with low latency', async () => {
      mockAxios.onPost('/api/v1/control').reply(200, {
        id: '789',
        status: 'SUCCESS'
      });

      const startTime = Date.now();
      const response = await apiService.sendControl(validControl);
      const duration = Date.now() - startTime;

      expect(response.id).toBe('789');
      expect(duration).toBeLessThan(SYSTEM_LIMITS.MAX_CONTROL_LATENCY);
    });

    it('should handle concurrent control requests', async () => {
      mockAxios.onPost('/api/v1/control').reply(200, {
        id: '789',
        status: 'SUCCESS'
      });

      const requests = Array(10).fill(validControl).map(apiService.sendControl);
      const responses = await Promise.all(requests);

      responses.forEach(response => {
        expect(response.status).toBe('SUCCESS');
      });
    });
  });

  describe('WebSocket Communication', () => {
    it('should establish WebSocket connection successfully', async () => {
      const connectPromise = new Promise<void>(resolve => {
        mockWebSocketServer.on('connection', () => resolve());
      });

      await apiService.setupWebSocket();
      await connectPromise;

      expect(apiService['wsConnection']?.readyState).toBe(WebSocket.OPEN);
    });

    it('should handle reconnection on connection loss', async () => {
      await apiService.setupWebSocket();
      mockWebSocketServer.close();

      // Fast-forward through reconnection attempts
      for (let i = 0; i < WEBSOCKET_CONFIG.MAX_RETRIES; i++) {
        jest.advanceTimersByTime(WEBSOCKET_CONFIG.RECONNECT_INTERVAL);
      }

      expect(apiService['reconnectAttempts']).toBe(WEBSOCKET_CONFIG.MAX_RETRIES);
    });

    it('should maintain heartbeat', async () => {
      await apiService.setupWebSocket();
      
      let heartbeatReceived = false;
      mockWebSocketServer.on('message', (data) => {
        const message = JSON.parse(data.toString());
        if (message.type === 'ping') heartbeatReceived = true;
      });

      jest.advanceTimersByTime(WEBSOCKET_CONFIG.PING_INTERVAL);
      expect(heartbeatReceived).toBe(true);
    });
  });

  describe('getStatus', () => {
    it('should retrieve system status with metrics', async () => {
      const mockStatus = {
        system_status: SystemHealthStatus.HEALTHY,
        resource_metrics: {
          cpu_usage_percent: 45,
          memory_usage_percent: 60,
          gpu_usage_percent: 75
        },
        performance_metrics: {
          generation_latency_ms: 80,
          frame_rate_fps: 24,
          control_response_ms: 45
        },
        timestamp: new Date().toISOString()
      };

      mockAxios.onGet('/api/v1/status').reply(200, mockStatus);

      const response = await apiService.getStatus();
      expect(response).toEqual(mockStatus);
      expect(response.performance_metrics.frame_rate_fps).toBeGreaterThanOrEqual(SYSTEM_LIMITS.MIN_FRAME_RATE);
    });

    it('should detect performance degradation', async () => {
      const degradedStatus = {
        system_status: SystemHealthStatus.DEGRADED,
        performance_metrics: {
          generation_latency_ms: 150,
          frame_rate_fps: 20
        }
      };

      mockAxios.onGet('/api/v1/status').reply(200, degradedStatus);

      const response = await apiService.getStatus();
      expect(response.system_status).toBe(SystemHealthStatus.DEGRADED);
    });
  });
});