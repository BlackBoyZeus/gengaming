/**
 * Enhanced React hook for managing video playback state, controls, and real-time streaming
 * Implements comprehensive frame buffering and performance optimization for 24 FPS at 720p
 * @version 1.0.0
 */

import { useState, useEffect, useCallback, useRef } from 'react'; // ^18.0.0
import { Video, VideoMetrics } from '../types/video';
import { VideoService } from '../services/video';
import { GenerationStatus } from '../types/api';

// Constants from technical specifications
const TARGET_FPS = 24;
const FRAME_BUFFER_SIZE = 102;
const MAX_LATENCY_MS = 100;
const CONTROL_DEBOUNCE_MS = 50;
const MIN_BUFFER_THRESHOLD = 24;

/**
 * Interface for video playback state with comprehensive metrics
 */
interface VideoState {
  isPlaying: boolean;
  currentFrame: number;
  totalFrames: number;
  frameRate: number;
  latency: number;
  bufferHealth: number;
  error: Error | null;
  status: GenerationStatus;
  metrics: VideoMetrics;
}

/**
 * Interface for video playback options
 */
interface VideoOptions {
  autoPlay?: boolean;
  loop?: boolean;
  preloadFrames?: number;
  onError?: (error: Error) => void;
  onMetricsUpdate?: (metrics: VideoMetrics) => void;
}

/**
 * Enhanced hook for video playback management with performance optimization
 */
export function useVideo(videoService: VideoService, options: VideoOptions = {}) {
  // State management with comprehensive metrics
  const [videoState, setVideoState] = useState<VideoState>({
    isPlaying: false,
    currentFrame: 0,
    totalFrames: 0,
    frameRate: 0,
    latency: 0,
    bufferHealth: 0,
    error: null,
    status: GenerationStatus.PENDING,
    metrics: {} as VideoMetrics
  });

  // Performance optimization refs
  const frameRequestRef = useRef<number>();
  const lastFrameTimeRef = useRef<number>(0);
  const frameBufferRef = useRef<Blob[]>([]);
  const metricsIntervalRef = useRef<NodeJS.Timer>();

  /**
   * Frame processing with performance optimization
   */
  const processFrame = useCallback((timestamp: number) => {
    if (!videoState.isPlaying) return;

    const elapsed = timestamp - lastFrameTimeRef.current;
    const frameInterval = 1000 / TARGET_FPS;

    if (elapsed >= frameInterval) {
      if (frameBufferRef.current.length > MIN_BUFFER_THRESHOLD) {
        const nextFrame = frameBufferRef.current.shift();
        if (nextFrame) {
          // Update frame and metrics
          setVideoState(prev => ({
            ...prev,
            currentFrame: prev.currentFrame + 1,
            frameRate: 1000 / elapsed,
            bufferHealth: frameBufferRef.current.length / FRAME_BUFFER_SIZE
          }));
        }
      }
      lastFrameTimeRef.current = timestamp;
    }

    frameRequestRef.current = requestAnimationFrame(processFrame);
  }, [videoState.isPlaying]);

  /**
   * Initialize video playback with error handling
   */
  useEffect(() => {
    const initializeVideo = async () => {
      try {
        const video = await videoService.getCurrentVideo();
        if (video) {
          setVideoState(prev => ({
            ...prev,
            totalFrames: video.frame_count,
            status: video.status,
            metrics: video.metrics
          }));

          if (options.autoPlay) {
            play();
          }
        }
      } catch (error) {
        handleError(error as Error);
      }
    };

    initializeVideo();
    return () => {
      if (frameRequestRef.current) {
        cancelAnimationFrame(frameRequestRef.current);
      }
      if (metricsIntervalRef.current) {
        clearInterval(metricsIntervalRef.current);
      }
    };
  }, []);

  /**
   * Frame buffer management with performance monitoring
   */
  useEffect(() => {
    const bufferFrames = async () => {
      try {
        const frameBuffer = await videoService.getFrameBuffer();
        frameBufferRef.current = frameBuffer;
        
        setVideoState(prev => ({
          ...prev,
          bufferHealth: frameBuffer.length / FRAME_BUFFER_SIZE
        }));
      } catch (error) {
        handleError(error as Error);
      }
    };

    bufferFrames();
  }, [videoState.currentFrame]);

  /**
   * Metrics collection and monitoring
   */
  useEffect(() => {
    metricsIntervalRef.current = setInterval(() => {
      const metrics = videoService.getMetrics();
      if (options.onMetricsUpdate) {
        options.onMetricsUpdate(metrics);
      }
    }, 1000);

    return () => {
      if (metricsIntervalRef.current) {
        clearInterval(metricsIntervalRef.current);
      }
    };
  }, [options.onMetricsUpdate]);

  /**
   * Error handling with recovery attempts
   */
  const handleError = useCallback((error: Error) => {
    setVideoState(prev => ({ ...prev, error }));
    if (options.onError) {
      options.onError(error);
    }
  }, [options.onError]);

  /**
   * Playback control functions with performance optimization
   */
  const play = useCallback(() => {
    if (frameBufferRef.current.length >= MIN_BUFFER_THRESHOLD) {
      setVideoState(prev => ({ ...prev, isPlaying: true }));
      lastFrameTimeRef.current = performance.now();
      frameRequestRef.current = requestAnimationFrame(processFrame);
    }
  }, [processFrame]);

  const pause = useCallback(() => {
    setVideoState(prev => ({ ...prev, isPlaying: false }));
    if (frameRequestRef.current) {
      cancelAnimationFrame(frameRequestRef.current);
    }
  }, []);

  const seek = useCallback((frame: number) => {
    if (frame >= 0 && frame < videoState.totalFrames) {
      setVideoState(prev => ({ ...prev, currentFrame: frame }));
    }
  }, [videoState.totalFrames]);

  const clear = useCallback(() => {
    videoService.clearVideo();
    setVideoState(prev => ({
      ...prev,
      isPlaying: false,
      currentFrame: 0,
      totalFrames: 0,
      error: null,
      status: GenerationStatus.PENDING
    }));
  }, []);

  return {
    videoState,
    play,
    pause,
    seek,
    clear,
    metrics: videoState.metrics
  };
}

export type { VideoState, VideoOptions };