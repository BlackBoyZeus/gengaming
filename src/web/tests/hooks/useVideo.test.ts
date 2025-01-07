import { renderHook, act } from '@testing-library/react-hooks';
import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { faker } from '@faker-js/faker';

import { useVideo } from '../../src/hooks/useVideo';
import { VideoService } from '../../src/services/video';
import { GenerationStatus } from '../../src/types/api';
import { VideoFormat } from '../../src/types/video';

// Constants from technical specifications
const TARGET_FPS = 24;
const FRAME_BUFFER_SIZE = 102;
const GENERATION_LATENCY_LIMIT = 100;
const CONTROL_LATENCY_LIMIT = 50;

// Mock VideoService implementation
jest.mock('../../src/services/video');

describe('useVideo Hook', () => {
  let mockVideoService: jest.Mocked<VideoService>;
  
  beforeEach(() => {
    // Reset all mocks before each test
    jest.clearAllMocks();
    
    // Setup VideoService mock with performance monitoring
    mockVideoService = {
      generateVideo: jest.fn().mockResolvedValue({
        id: faker.string.uuid(),
        generation_id: faker.string.uuid(),
        status: GenerationStatus.COMPLETED,
        format: VideoFormat.MP4,
        frame_count: FRAME_BUFFER_SIZE,
        metrics: {
          frame_rate: TARGET_FPS,
          resolution_width: 1280,
          resolution_height: 720,
          duration_seconds: FRAME_BUFFER_SIZE / TARGET_FPS,
          latency_ms: 95,
          quality_score: 85,
          buffer_size: 1024 * 1024,
          fid_score: 250,
          fvd_score: 850
        }
      }),
      processFrame: jest.fn().mockImplementation((blob, sequence) => 
        Promise.resolve({
          id: `frame-${sequence}`,
          sequence,
          data: blob,
          timestamp: Date.now(),
          metadata: {},
          validation_status: true
        })
      ),
      getFrameBuffer: jest.fn().mockReturnValue({
        length: FRAME_BUFFER_SIZE,
        frames: Array(FRAME_BUFFER_SIZE).fill(null).map((_, i) => ({
          id: `frame-${i}`,
          sequence: i,
          data: new Blob(),
          timestamp: Date.now() - (1000 / TARGET_FPS) * i,
          metadata: {},
          validation_status: true
        }))
      }),
      getMetrics: jest.fn().mockReturnValue({
        currentFps: TARGET_FPS,
        averageLatency: 95,
        bufferHealth: 1.0,
        frameDrops: 0,
        processingTime: 45,
        memoryUsage: 512 * 1024 * 1024
      })
    } as unknown as jest.Mocked<VideoService>;
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should initialize with correct default state', () => {
    const { result } = renderHook(() => useVideo(mockVideoService));

    expect(result.current.videoState).toEqual({
      isPlaying: false,
      currentFrame: 0,
      totalFrames: 0,
      frameRate: 0,
      latency: 0,
      bufferHealth: 0,
      error: null,
      status: GenerationStatus.PENDING,
      metrics: {}
    });
  });

  it('should maintain target frame rate of 24 FPS', async () => {
    jest.useFakeTimers();
    
    const { result } = renderHook(() => useVideo(mockVideoService));
    
    // Start playback
    await act(async () => {
      result.current.play();
      // Advance timers to simulate frame processing
      for (let i = 0; i < 24; i++) {
        jest.advanceTimersByTime(1000 / TARGET_FPS);
      }
    });

    expect(result.current.videoState.frameRate).toBeCloseTo(TARGET_FPS, 1);
    expect(mockVideoService.processFrame).toHaveBeenCalledTimes(24);
  });

  it('should meet latency requirements for generation and control', async () => {
    const { result } = renderHook(() => useVideo(mockVideoService));
    
    const startTime = performance.now();
    
    await act(async () => {
      await mockVideoService.generateVideo('test prompt', {
        resolution: { width: 1280, height: 720 },
        frames: FRAME_BUFFER_SIZE,
        perspective: 'THIRD_PERSON',
        fps: TARGET_FPS
      });
    });

    const generationLatency = performance.now() - startTime;
    expect(generationLatency).toBeLessThan(GENERATION_LATENCY_LIMIT);

    // Test control response time
    const controlStartTime = performance.now();
    await act(async () => {
      result.current.seek(50);
    });

    const controlLatency = performance.now() - controlStartTime;
    expect(controlLatency).toBeLessThan(CONTROL_LATENCY_LIMIT);
  });

  it('should maintain healthy frame buffer', async () => {
    const { result } = renderHook(() => useVideo(mockVideoService));

    await act(async () => {
      result.current.play();
    });

    expect(result.current.videoState.bufferHealth).toBeGreaterThanOrEqual(0.8);
    expect(mockVideoService.getFrameBuffer().length).toBe(FRAME_BUFFER_SIZE);
  });

  it('should handle playback controls correctly', async () => {
    const { result } = renderHook(() => useVideo(mockVideoService));

    // Test play
    await act(async () => {
      result.current.play();
    });
    expect(result.current.videoState.isPlaying).toBe(true);

    // Test pause
    await act(async () => {
      result.current.pause();
    });
    expect(result.current.videoState.isPlaying).toBe(false);

    // Test seek
    await act(async () => {
      result.current.seek(50);
    });
    expect(result.current.videoState.currentFrame).toBe(50);
  });

  it('should handle error recovery gracefully', async () => {
    const { result } = renderHook(() => useVideo(mockVideoService, {
      onError: jest.fn()
    }));

    // Simulate frame processing error
    mockVideoService.processFrame.mockRejectedValueOnce(new Error('Frame processing failed'));

    await act(async () => {
      result.current.play();
    });

    expect(result.current.videoState.error).toBeTruthy();
    expect(result.current.videoState.isPlaying).toBe(false);

    // Test recovery
    mockVideoService.processFrame.mockResolvedValueOnce({
      id: 'recovery-frame',
      sequence: 0,
      data: new Blob(),
      timestamp: Date.now(),
      metadata: {},
      validation_status: true
    });

    await act(async () => {
      result.current.play();
    });

    expect(result.current.videoState.error).toBeNull();
    expect(result.current.videoState.isPlaying).toBe(true);
  });

  it('should track and report performance metrics', async () => {
    const { result } = renderHook(() => useVideo(mockVideoService, {
      onMetricsUpdate: jest.fn()
    }));

    await act(async () => {
      result.current.play();
      // Let metrics collection cycle run
      jest.advanceTimersByTime(1000);
    });

    const metrics = result.current.videoState.metrics;
    expect(metrics.frame_rate).toBe(TARGET_FPS);
    expect(metrics.latency_ms).toBeLessThan(GENERATION_LATENCY_LIMIT);
    expect(metrics.resolution_width).toBe(1280);
    expect(metrics.resolution_height).toBe(720);
  });
});