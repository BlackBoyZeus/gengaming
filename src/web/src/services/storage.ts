/**
 * Browser storage service for GameGen-X web interface
 * Manages video data, generation states, and user preferences
 * @version 1.0.0
 */

import { Video, VideoFormat } from '../types/video';
import { GenerationStatus } from '../types/api';

// Storage keys for different data types
const STORAGE_KEYS = {
  VIDEO_CACHE: 'gamegen-x:video-cache',
  USER_PREFERENCES: 'gamegen-x:preferences',
  GENERATION_STATES: 'gamegen-x:generation-states',
  SESSION_DATA: 'gamegen-x:session'
} as const;

// Storage configuration and limits
const STORAGE_CONFIG = {
  MAX_VIDEO_CACHE_SIZE: 52428800, // 50MB in bytes
  MAX_CACHE_AGE_HOURS: 24,
  CLEANUP_THRESHOLD_PERCENT: 80,
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY_MS: 1000
} as const;

/**
 * Custom error types for storage operations
 */
class StorageQuotaError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'StorageQuotaError';
  }
}

class StorageValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'StorageValidationError';
  }
}

/**
 * Type definitions for storage entries
 */
interface CacheEntry<T> {
  data: T;
  timestamp: number;
  expirationTime: number;
}

interface StorageMetrics {
  totalSize: number;
  itemCount: number;
  lastCleanup: number;
}

/**
 * Storage service implementation
 */
class StorageServiceImpl {
  private metrics: StorageMetrics = {
    totalSize: 0,
    itemCount: 0,
    lastCleanup: Date.now()
  };

  /**
   * Stores video data in localStorage with expiration
   */
  async setVideoCache(video: Video, expirationHours: number = STORAGE_CONFIG.MAX_CACHE_AGE_HOURS): Promise<void> {
    try {
      const entry: CacheEntry<Video> = {
        data: video,
        timestamp: Date.now(),
        expirationTime: Date.now() + (expirationHours * 60 * 60 * 1000)
      };

      const serializedEntry = JSON.stringify(entry);
      const entrySize = new Blob([serializedEntry]).size;

      if (entrySize > STORAGE_CONFIG.MAX_VIDEO_CACHE_SIZE) {
        throw new StorageQuotaError('Video data exceeds maximum cache size');
      }

      await this.checkStorageQuota(entrySize);
      localStorage.setItem(`${STORAGE_KEYS.VIDEO_CACHE}:${video.id}`, serializedEntry);
      
      this.updateMetrics('add', entrySize);
      await this.cleanupIfNeeded();
    } catch (error) {
      if (error instanceof StorageQuotaError) {
        throw error;
      }
      throw new Error(`Failed to cache video: ${error.message}`);
    }
  }

  /**
   * Retrieves cached video data with validation
   */
  async getVideoCache(videoId: string): Promise<Video | null> {
    try {
      const serializedEntry = localStorage.getItem(`${STORAGE_KEYS.VIDEO_CACHE}:${videoId}`);
      if (!serializedEntry) return null;

      const entry: CacheEntry<Video> = JSON.parse(serializedEntry);
      
      if (Date.now() > entry.expirationTime) {
        await this.removeVideoCache(videoId);
        return null;
      }

      this.validateVideoData(entry.data);
      return entry.data;
    } catch (error) {
      console.error(`Error retrieving video cache: ${error.message}`);
      return null;
    }
  }

  /**
   * Stores user preferences with validation
   */
  async setUserPreferences(preferences: Record<string, any>): Promise<void> {
    try {
      const existing = await this.getUserPreferences();
      const merged = { ...existing, ...preferences, lastModified: Date.now() };
      localStorage.setItem(STORAGE_KEYS.USER_PREFERENCES, JSON.stringify(merged));
    } catch (error) {
      throw new Error(`Failed to store preferences: ${error.message}`);
    }
  }

  /**
   * Retrieves user preferences with defaults
   */
  async getUserPreferences(): Promise<Record<string, any>> {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.USER_PREFERENCES);
      return stored ? JSON.parse(stored) : this.getDefaultPreferences();
    } catch (error) {
      console.error(`Error retrieving preferences: ${error.message}`);
      return this.getDefaultPreferences();
    }
  }

  /**
   * Manages generation state with cleanup
   */
  async setGenerationState(
    generationId: string,
    status: GenerationStatus,
    metadata: Record<string, any>
  ): Promise<void> {
    try {
      const state = {
        status,
        metadata,
        timestamp: Date.now()
      };
      sessionStorage.setItem(
        `${STORAGE_KEYS.GENERATION_STATES}:${generationId}`,
        JSON.stringify(state)
      );
      await this.cleanupGenerationStates();
    } catch (error) {
      throw new Error(`Failed to store generation state: ${error.message}`);
    }
  }

  /**
   * Retrieves generation state with validation
   */
  async getGenerationState(generationId: string): Promise<{ status: GenerationStatus; metadata: Record<string, any> } | null> {
    try {
      const stored = sessionStorage.getItem(`${STORAGE_KEYS.GENERATION_STATES}:${generationId}`);
      if (!stored) return null;

      const state = JSON.parse(stored);
      if (!Object.values(GenerationStatus).includes(state.status)) {
        throw new StorageValidationError('Invalid generation status');
      }

      return {
        status: state.status,
        metadata: state.metadata
      };
    } catch (error) {
      console.error(`Error retrieving generation state: ${error.message}`);
      return null;
    }
  }

  /**
   * Performs comprehensive storage cleanup
   */
  async clearStorage(): Promise<void> {
    try {
      const preferences = await this.getUserPreferences();
      
      localStorage.clear();
      sessionStorage.clear();
      
      await this.setUserPreferences(preferences);
      this.resetMetrics();
      
      console.log('Storage cleared successfully');
    } catch (error) {
      throw new Error(`Failed to clear storage: ${error.message}`);
    }
  }

  /**
   * Private helper methods
   */
  private async checkStorageQuota(requiredSize: number): Promise<void> {
    if (this.metrics.totalSize + requiredSize > STORAGE_CONFIG.MAX_VIDEO_CACHE_SIZE) {
      await this.cleanup();
    }
  }

  private async cleanupIfNeeded(): Promise<void> {
    const threshold = (STORAGE_CONFIG.MAX_VIDEO_CACHE_SIZE * STORAGE_CONFIG.CLEANUP_THRESHOLD_PERCENT) / 100;
    if (this.metrics.totalSize > threshold) {
      await this.cleanup();
    }
  }

  private async cleanup(): Promise<void> {
    const now = Date.now();
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith(`${STORAGE_KEYS.VIDEO_CACHE}:`)) {
        const entry: CacheEntry<Video> = JSON.parse(localStorage.getItem(key) || '');
        if (now > entry.expirationTime) {
          localStorage.removeItem(key);
          this.updateMetrics('remove', new Blob([JSON.stringify(entry)]).size);
        }
      }
    }
    this.metrics.lastCleanup = now;
  }

  private async cleanupGenerationStates(): Promise<void> {
    const now = Date.now();
    const expirationTime = now - (STORAGE_CONFIG.MAX_CACHE_AGE_HOURS * 60 * 60 * 1000);

    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key?.startsWith(`${STORAGE_KEYS.GENERATION_STATES}:`)) {
        const state = JSON.parse(sessionStorage.getItem(key) || '');
        if (state.timestamp < expirationTime) {
          sessionStorage.removeItem(key);
        }
      }
    }
  }

  private async removeVideoCache(videoId: string): Promise<void> {
    const key = `${STORAGE_KEYS.VIDEO_CACHE}:${videoId}`;
    const entry = localStorage.getItem(key);
    if (entry) {
      this.updateMetrics('remove', new Blob([entry]).size);
      localStorage.removeItem(key);
    }
  }

  private validateVideoData(video: Video): void {
    if (!video.id || !video.metrics || typeof video.frame_count !== 'number') {
      throw new StorageValidationError('Invalid video data structure');
    }
  }

  private getDefaultPreferences(): Record<string, any> {
    return {
      theme: 'light',
      videoFormat: VideoFormat.MP4,
      autoplay: true,
      lastModified: Date.now()
    };
  }

  private updateMetrics(operation: 'add' | 'remove', size: number): void {
    if (operation === 'add') {
      this.metrics.totalSize += size;
      this.metrics.itemCount++;
    } else {
      this.metrics.totalSize = Math.max(0, this.metrics.totalSize - size);
      this.metrics.itemCount = Math.max(0, this.metrics.itemCount - 1);
    }
  }

  private resetMetrics(): void {
    this.metrics = {
      totalSize: 0,
      itemCount: 0,
      lastCleanup: Date.now()
    };
  }
}

// Export singleton instance
export const StorageService = new StorageServiceImpl();