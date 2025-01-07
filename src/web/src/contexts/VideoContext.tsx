/**
 * VideoContext - React Context provider for GameGen-X video management
 * @version 1.0.0
 * 
 * Implements comprehensive video state management, real-time frame updates,
 * and performance monitoring while maintaining 24 FPS at 720p resolution.
 */

import React, { createContext, useContext, useCallback, useEffect, useState, useRef } from 'react'; // ^18.0.0
import { Video, VideoFrame } from '../types/video';
import { VideoService } from '../services/video';
import { calculateFrameMetrics } from '../utils/video';
import { GenerationStatus } from '../types/generation';

// Performance constants from technical specifications
const TARGET_FPS = 24;
const MAX_LATENCY_MS = 100;
const MAX_BUFFER_SIZE = 102;
const RECONNECT_TIMEOUT = 5000;

/**
 * Interface for video context performance metrics
 */
interface PerformanceMetrics {
  currentFps: number;
  averageLatency: number;
  bufferHealth: number;
  memoryUsage: number;
  frameDrops: number;
}

/**
 * Interface for WebSocket connection state
 */
interface WebSocketState {
  connected: boolean;
  reconnecting: boolean;
  error: Error | null;
}

/**
 * Interface for video context value
 */
interface VideoContextValue {
  currentVideo: Video | null;
  frameBuffer: VideoFrame[];
  performanceMetrics: PerformanceMetrics;
  connectionState: WebSocketState;
  generateVideo: (prompt: string) => Promise<void>;
  clearVideo: () => void;
  isGenerating: boolean;
  error: Error | null;
}

// Create context with comprehensive type safety
const VideoContext = createContext<VideoContextValue | null>(null);

/**
 * VideoProvider component implementing comprehensive video management
 */
export const VideoProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Core state management
  const [currentVideo, setCurrentVideo] = useState<Video | null>(null);
  const [frameBuffer, setFrameBuffer] = useState<VideoFrame[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // Performance monitoring state
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
    currentFps: 0,
    averageLatency: 0,
    bufferHealth: 1,
    memoryUsage: 0,
    frameDrops: 0
  });

  // WebSocket connection state
  const [connectionState, setConnectionState] = useState<WebSocketState>({
    connected: false,
    reconnecting: false,
    error: null
  });

  // Service and refs
  const videoService = useRef(new VideoService());
  const frameCountRef = useRef(0);
  const lastFrameTimeRef = useRef(performance.now());

  /**
   * Initialize video service and setup event listeners
   */
  useEffect(() => {
    const service = videoService.current;

    service.eventEmitter.on('frame', handleFrameUpdate);
    service.eventEmitter.on('metrics', handleMetricsUpdate);
    service.eventEmitter.on('error', handleServiceError);
    service.eventEmitter.on('connectionState', handleConnectionState);

    return () => {
      service.dispose();
      service.eventEmitter.removeAllListeners();
    };
  }, []);

  /**
   * Handle real-time frame updates with performance optimization
   */
  const handleFrameUpdate = useCallback((frame: VideoFrame) => {
    const now = performance.now();
    const timeDelta = now - lastFrameTimeRef.current;
    
    setFrameBuffer(prevBuffer => {
      // Optimize buffer size
      if (prevBuffer.length >= MAX_BUFFER_SIZE) {
        prevBuffer = prevBuffer.slice(1);
      }
      return [...prevBuffer, frame];
    });

    // Update performance metrics
    frameCountRef.current++;
    if (timeDelta >= 1000) {
      const metrics = calculateFrameMetrics(frame, now, frameBuffer.length);
      setPerformanceMetrics(prev => ({
        ...prev,
        currentFps: (frameCountRef.current * 1000) / timeDelta,
        averageLatency: metrics.latency_ms,
        bufferHealth: frameBuffer.length / MAX_BUFFER_SIZE,
        memoryUsage: metrics.buffer_size,
      }));
      
      frameCountRef.current = 0;
      lastFrameTimeRef.current = now;
    }
  }, [frameBuffer.length]);

  /**
   * Handle performance metrics updates
   */
  const handleMetricsUpdate = useCallback((metrics: PerformanceMetrics) => {
    setPerformanceMetrics(metrics);
  }, []);

  /**
   * Handle service errors with recovery attempts
   */
  const handleServiceError = useCallback((error: Error) => {
    setError(error);
    setConnectionState(prev => ({
      ...prev,
      error,
      connected: false
    }));
  }, []);

  /**
   * Handle WebSocket connection state changes
   */
  const handleConnectionState = useCallback((state: WebSocketState) => {
    setConnectionState(state);
    if (!state.connected && !state.reconnecting) {
      setTimeout(() => {
        videoService.current.reconnectWebSocket();
      }, RECONNECT_TIMEOUT);
    }
  }, []);

  /**
   * Generate new video with comprehensive error handling
   */
  const generateVideo = useCallback(async (prompt: string) => {
    try {
      setIsGenerating(true);
      setError(null);
      
      const video = await videoService.current.generateVideo(prompt, {
        resolution: { width: 1280, height: 720 },
        frames: MAX_BUFFER_SIZE,
        fps: TARGET_FPS,
        perspective: 'THIRD_PERSON'
      });

      setCurrentVideo(video);
      setFrameBuffer([]);
      frameCountRef.current = 0;
      lastFrameTimeRef.current = performance.now();
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Video generation failed'));
    } finally {
      setIsGenerating(false);
    }
  }, []);

  /**
   * Clear video state with resource cleanup
   */
  const clearVideo = useCallback(() => {
    setCurrentVideo(null);
    setFrameBuffer([]);
    setError(null);
    frameCountRef.current = 0;
    videoService.current.cleanupFrameBuffer();
  }, []);

  // Construct context value with memoization
  const contextValue = React.useMemo<VideoContextValue>(() => ({
    currentVideo,
    frameBuffer,
    performanceMetrics,
    connectionState,
    generateVideo,
    clearVideo,
    isGenerating,
    error
  }), [
    currentVideo,
    frameBuffer,
    performanceMetrics,
    connectionState,
    generateVideo,
    clearVideo,
    isGenerating,
    error
  ]);

  return (
    <VideoContext.Provider value={contextValue}>
      {children}
    </VideoContext.Provider>
  );
};

/**
 * Custom hook for accessing video context with type safety
 */
export const useVideoContext = (): VideoContextValue => {
  const context = useContext(VideoContext);
  if (!context) {
    throw new Error('useVideoContext must be used within a VideoProvider');
  }
  return context;
};

export default VideoContext;