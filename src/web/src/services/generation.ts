/**
 * GameGen-X Generation Service
 * @version 1.0.0
 * 
 * Service module for managing video generation operations with quality metrics
 * and progress tracking in the GameGen-X web interface.
 */

import { EventEmitter } from 'events'; // v3.3.0
import APIService from './api';
import { API_ENDPOINTS } from '../config/api';
import {
  GenerationParameters,
  GenerationState,
  GenerationStatus,
  GenerationMetrics,
  MAX_FID_SCORE,
  MAX_FVD_SCORE,
  TARGET_FPS
} from '../types/generation';

/**
 * Event types for generation progress tracking
 */
enum GenerationEventType {
  PROGRESS = 'progress',
  METRICS = 'metrics',
  ERROR = 'error',
  COMPLETE = 'complete'
}

/**
 * Enhanced service class for managing video generation operations
 */
export class GenerationService {
  private apiService: APIService;
  private eventEmitter: EventEmitter;
  private currentGeneration: GenerationState | null;
  private metricsInterval: NodeJS.Timer | null;
  private readonly METRICS_POLL_INTERVAL = 1000; // 1 second
  private readonly GENERATION_TIMEOUT = 300000; // 5 minutes

  constructor(apiService: APIService) {
    this.apiService = apiService;
    this.eventEmitter = new EventEmitter();
    this.currentGeneration = null;
    this.metricsInterval = null;
  }

  /**
   * Initiates a new video generation process with enhanced validation
   */
  public async startGeneration(
    prompt: string,
    parameters: GenerationParameters
  ): Promise<GenerationState> {
    // Validate input parameters
    this.validateGenerationParameters(parameters);
    this.validatePrompt(prompt);

    try {
      // Initialize generation request
      const response = await this.apiService.generateVideo({
        prompt,
        parameters,
        timestamp: new Date().toISOString()
      });

      // Initialize generation state
      this.currentGeneration = {
        id: response.id,
        prompt,
        parameters,
        status: GenerationStatus.GENERATING,
        progress: 0,
        metrics: {
          fid_score: 0,
          fvd_score: 0,
          generation_time_ms: 0,
          actual_fps: 0
        },
        error: null
      };

      // Start metrics monitoring
      this.startMetricsMonitoring();

      return this.currentGeneration;
    } catch (error) {
      throw new Error(`Generation initialization failed: ${error.message}`);
    }
  }

  /**
   * Retrieves current generation state with latest metrics
   */
  public getGenerationState(): GenerationState | null {
    return this.currentGeneration;
  }

  /**
   * Cancels ongoing generation process with cleanup
   */
  public async cancelGeneration(): Promise<void> {
    if (!this.currentGeneration) {
      return;
    }

    try {
      await this.apiService.cancelGeneration(this.currentGeneration.id);
      this.stopMetricsMonitoring();
      this.emitGenerationEvent(GenerationEventType.COMPLETE, {
        status: GenerationStatus.FAILED,
        error: 'Generation cancelled by user'
      });
      this.resetState();
    } catch (error) {
      throw new Error(`Failed to cancel generation: ${error.message}`);
    }
  }

  /**
   * Registers progress update callback with enhanced event typing
   */
  public onProgress(callback: (state: GenerationState) => void): void {
    this.eventEmitter.on(GenerationEventType.PROGRESS, callback);
  }

  /**
   * Retrieves current generation quality metrics
   */
  public async getGenerationMetrics(): Promise<GenerationMetrics> {
    if (!this.currentGeneration) {
      throw new Error('No active generation');
    }

    try {
      const metrics = await this.apiService.getGenerationMetrics(this.currentGeneration.id);
      this.validateMetrics(metrics);
      
      if (this.currentGeneration) {
        this.currentGeneration.metrics = metrics;
      }

      return metrics;
    } catch (error) {
      throw new Error(`Failed to fetch metrics: ${error.message}`);
    }
  }

  /**
   * Validates generation parameters against system requirements
   */
  private validateGenerationParameters(parameters: GenerationParameters): void {
    if (parameters.fps < TARGET_FPS) {
      throw new Error(`Frame rate must be at least ${TARGET_FPS} FPS`);
    }

    if (parameters.frames < 1) {
      throw new Error('Frame count must be positive');
    }

    if (parameters.resolution.width < 1 || parameters.resolution.height < 1) {
      throw new Error('Invalid resolution dimensions');
    }
  }

  /**
   * Validates generation prompt
   */
  private validatePrompt(prompt: string): void {
    if (!prompt || prompt.trim().length === 0) {
      throw new Error('Prompt cannot be empty');
    }
  }

  /**
   * Validates generation metrics against quality requirements
   */
  private validateMetrics(metrics: GenerationMetrics): void {
    if (metrics.fid_score > MAX_FID_SCORE) {
      this.emitGenerationEvent(GenerationEventType.ERROR, {
        error: `FID score ${metrics.fid_score} exceeds maximum threshold ${MAX_FID_SCORE}`
      });
    }

    if (metrics.fvd_score > MAX_FVD_SCORE) {
      this.emitGenerationEvent(GenerationEventType.ERROR, {
        error: `FVD score ${metrics.fvd_score} exceeds maximum threshold ${MAX_FVD_SCORE}`
      });
    }

    if (metrics.actual_fps < TARGET_FPS) {
      this.emitGenerationEvent(GenerationEventType.ERROR, {
        error: `Actual FPS ${metrics.actual_fps} below target ${TARGET_FPS}`
      });
    }
  }

  /**
   * Starts metrics monitoring interval
   */
  private startMetricsMonitoring(): void {
    this.metricsInterval = setInterval(async () => {
      try {
        const metrics = await this.getGenerationMetrics();
        this.emitGenerationEvent(GenerationEventType.METRICS, { metrics });
      } catch (error) {
        console.error('Metrics monitoring error:', error);
      }
    }, this.METRICS_POLL_INTERVAL);

    // Set generation timeout
    setTimeout(() => {
      if (this.currentGeneration?.status === GenerationStatus.GENERATING) {
        this.cancelGeneration();
      }
    }, this.GENERATION_TIMEOUT);
  }

  /**
   * Stops metrics monitoring and cleans up
   */
  private stopMetricsMonitoring(): void {
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
      this.metricsInterval = null;
    }
  }

  /**
   * Emits generation events with type safety
   */
  private emitGenerationEvent(type: GenerationEventType, data: any): void {
    if (this.currentGeneration) {
      this.eventEmitter.emit(type, {
        ...this.currentGeneration,
        ...data
      });
    }
  }

  /**
   * Resets service state
   */
  private resetState(): void {
    this.currentGeneration = null;
    this.stopMetricsMonitoring();
  }
}

export default new GenerationService(APIService);