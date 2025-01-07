/**
 * GameGen-X Video Service
 * @version 1.0.0
 * 
 * Core video service module handling video generation, streaming, and real-time control
 * with comprehensive performance monitoring and error handling.
 */

import { EventEmitter } from 'events'; // v3.3.0
import { Video, VideoFrame, VideoMetrics } from '../types/video';
import { APIService } from './api';
import { GenerationParameters, GenerationStatus } from '../types/generation';
import { SYSTEM_LIMITS, VIDEO_SETTINGS, WEBSOCKET_CONFIG } from '../config/constants';

// Service constants aligned with technical specifications
const FRAME_BUFFER_SIZE = VIDEO_SETTINGS.DEFAULT_FRAME_COUNT;
const TARGET_FPS = SYSTEM_LIMITS.MIN_FRAME_RATE;
const MAX_LATENCY_MS = SYSTEM_LIMITS.MAX_CONTROL_LATENCY;
const METRICS_UPDATE_INTERVAL_MS = 500;
const MAX_RETRY_ATTEMPTS = 3;

/**
 * Interface for frame buffer management configuration
 */
interface BufferConfig {
  maxSize: number;
  preloadFrames: number;
  targetLatency: number;
}

/**
 * Interface for service performance metrics
 */
interface ServiceMetrics {
  currentFps: number;
  averageLatency: number;
  bufferHealth: number;
  frameDrops: number;
  processingTime: number;
  memoryUsage: number;
}

/**
 * Enum for connection state tracking
 */
enum ConnectionState {
  DISCONNECTED = 'DISCONNECTED',
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  RECONNECTING = 'RECONNECTING',
  ERROR = 'ERROR'
}

/**
 * Frame buffer management class for optimized video playback
 */
class BufferManager {
  private frames: VideoFrame[] = [];
  private config: BufferConfig;
  private metrics: { drops: number; latency: number[] } = { drops: 0, latency: [] };

  constructor(config: BufferConfig) {
    this.config = config;
  }

  addFrame(frame: VideoFrame): void {
    if (this.frames.length >= this.config.maxSize) {
      this.frames.shift();
      this.metrics.drops++;
    }
    this.frames.push(frame);
    this.updateMetrics(frame);
  }

  private updateMetrics(frame: VideoFrame): void {
    const latency = Date.now() - frame.timestamp;
    this.metrics.latency.push(latency);
    if (this.metrics.latency.length > 100) this.metrics.latency.shift();
  }

  getMetrics(): { bufferHealth: number; averageLatency: number; drops: number } {
    return {
      bufferHealth: this.frames.length / this.config.maxSize,
      averageLatency: this.metrics.latency.reduce((a, b) => a + b, 0) / this.metrics.latency.length,
      drops: this.metrics.drops
    };
  }
}

/**
 * Core video service class implementing comprehensive video management
 */
export class VideoService {
  private apiService: APIService;
  private currentVideo: Video | null = null;
  private bufferManager: BufferManager;
  private eventEmitter: EventEmitter;
  private metrics: ServiceMetrics;
  private connectionState: ConnectionState = ConnectionState.DISCONNECTED;
  private metricsInterval: NodeJS.Timer | null = null;

  constructor(apiService: APIService, bufferConfig?: Partial<BufferConfig>) {
    this.apiService = apiService;
    this.eventEmitter = new EventEmitter();
    this.bufferManager = new BufferManager({
      maxSize: FRAME_BUFFER_SIZE,
      preloadFrames: Math.ceil(FRAME_BUFFER_SIZE * 0.2),
      targetLatency: MAX_LATENCY_MS,
      ...bufferConfig
    });
    this.metrics = this.initializeMetrics();
    this.startMetricsCollection();
  }

  /**
   * Initialize service metrics tracking
   */
  private initializeMetrics(): ServiceMetrics {
    return {
      currentFps: 0,
      averageLatency: 0,
      bufferHealth: 1,
      frameDrops: 0,
      processingTime: 0,
      memoryUsage: 0
    };
  }

  /**
   * Start periodic metrics collection
   */
  private startMetricsCollection(): void {
    this.metricsInterval = setInterval(() => {
      const bufferMetrics = this.bufferManager.getMetrics();
      this.metrics = {
        ...this.metrics,
        bufferHealth: bufferMetrics.bufferHealth,
        averageLatency: bufferMetrics.averageLatency,
        frameDrops: bufferMetrics.drops
      };
      this.eventEmitter.emit('metrics', this.metrics);
    }, METRICS_UPDATE_INTERVAL_MS);
  }

  /**
   * Generate new video with comprehensive error handling
   */
  public async generateVideo(prompt: string, parameters: GenerationParameters): Promise<Video> {
    try {
      this.connectionState = ConnectionState.CONNECTING;
      this.eventEmitter.emit('connectionState', this.connectionState);

      const video = await this.apiService.generateVideo({
        prompt,
        parameters,
        timestamp: new Date().toISOString()
      });

      this.currentVideo = video;
      this.connectionState = ConnectionState.CONNECTED;
      this.eventEmitter.emit('connectionState', this.connectionState);
      this.eventEmitter.emit('videoGenerated', video);

      return video;
    } catch (error) {
      this.connectionState = ConnectionState.ERROR;
      this.eventEmitter.emit('connectionState', this.connectionState);
      this.eventEmitter.emit('error', error);
      throw error;
    }
  }

  /**
   * Process incoming video frame with performance optimization
   */
  public async processFrame(frameData: Blob, sequence: number, metrics: VideoMetrics): Promise<void> {
    const startTime = performance.now();

    try {
      const frame: VideoFrame = {
        id: `${this.currentVideo?.id}-${sequence}`,
        sequence,
        data: frameData,
        timestamp: Date.now(),
        metadata: {},
        validation_status: true
      };

      this.bufferManager.addFrame(frame);
      this.metrics.processingTime = performance.now() - startTime;
      this.eventEmitter.emit('frame', frame);
    } catch (error) {
      this.eventEmitter.emit('frameError', { sequence, error });
      throw error;
    }
  }

  /**
   * Get current service metrics
   */
  public getMetrics(): ServiceMetrics {
    return { ...this.metrics };
  }

  /**
   * Get current connection state
   */
  public getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  /**
   * Clean up service resources
   */
  public dispose(): void {
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
    }
    this.eventEmitter.removeAllListeners();
  }
}