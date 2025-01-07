/**
 * GameGen-X Generation Hook
 * @version 1.0.0
 * 
 * Custom React hook for managing video generation state and operations with comprehensive
 * metrics tracking and performance monitoring.
 */

import { useState, useCallback, useEffect, useRef } from 'react'; // ^18.0.0
import { useGenerationContext } from '../contexts/GenerationContext';
import { GenerationParameters, GenerationState } from '../types/generation';

/**
 * Interface defining the return type of the useGeneration hook
 */
interface UseGenerationResult {
  startGeneration: (prompt: string, parameters: GenerationParameters) => Promise<string>;
  cancelGeneration: () => Promise<void>;
  generationState: GenerationState | null;
  isGenerating: boolean;
  error: Error | null;
  progress: number;
  metrics: {
    fid_score: number;
    fvd_score: number;
    generation_time_ms: number;
    actual_fps: number;
  };
  performance: {
    generationLatency: number;
    controlLatency: number;
    frameRate: number;
    qualityScore: number;
  };
  quality: {
    isValid: boolean;
    warnings: string[];
  };
}

/**
 * Custom hook for managing video generation with comprehensive metrics tracking
 */
export const useGeneration = (): UseGenerationResult => {
  // Access generation context
  const {
    generationState,
    isGenerating,
    error: contextError,
    progress,
    metrics,
    performance,
    startGeneration: contextStartGeneration,
    cancelGeneration: contextCancelGeneration,
    validateQuality,
    monitorPerformance
  } = useGenerationContext();

  // Local state for enhanced error handling and quality tracking
  const [error, setError] = useState<Error | null>(null);
  const [qualityWarnings, setQualityWarnings] = useState<string[]>([]);
  const performanceRef = useRef<number[]>([]);
  const qualityCheckInterval = useRef<NodeJS.Timer>();

  /**
   * Enhanced generation start with parameter validation and metrics initialization
   */
  const startGeneration = useCallback(async (
    prompt: string,
    parameters: GenerationParameters
  ): Promise<string> => {
    try {
      setError(null);
      setQualityWarnings([]);
      performanceRef.current = [];

      // Validate generation parameters
      if (parameters.fps < 24) {
        throw new Error('Frame rate must be at least 24 FPS');
      }

      if (parameters.frames < 1) {
        throw new Error('Frame count must be positive');
      }

      // Start generation with context
      const generationId = await contextStartGeneration(prompt, parameters);

      // Initialize performance monitoring
      performanceRef.current.push(Date.now());

      return generationId;
    } catch (err) {
      const enhancedError = err instanceof Error ? err : new Error('Generation failed');
      setError(enhancedError);
      throw enhancedError;
    }
  }, [contextStartGeneration]);

  /**
   * Enhanced generation cancellation with cleanup
   */
  const cancelGeneration = useCallback(async (): Promise<void> => {
    try {
      await contextCancelGeneration();
      setError(null);
      setQualityWarnings([]);
      performanceRef.current = [];
    } catch (err) {
      const enhancedError = err instanceof Error ? err : new Error('Cancellation failed');
      setError(enhancedError);
      throw enhancedError;
    }
  }, [contextCancelGeneration]);

  /**
   * Monitor generation quality and update warnings
   */
  useEffect(() => {
    if (isGenerating) {
      qualityCheckInterval.current = setInterval(async () => {
        const isQualityValid = await validateQuality();
        
        const warnings: string[] = [];
        if (metrics.fid_score > 300) {
          warnings.push(`FID score (${metrics.fid_score}) exceeds maximum threshold (300)`);
        }
        if (metrics.fvd_score > 1000) {
          warnings.push(`FVD score (${metrics.fvd_score}) exceeds maximum threshold (1000)`);
        }
        if (metrics.actual_fps < 24) {
          warnings.push(`Frame rate (${metrics.actual_fps} FPS) below target (24 FPS)`);
        }

        setQualityWarnings(warnings);
      }, 1000);

      return () => {
        if (qualityCheckInterval.current) {
          clearInterval(qualityCheckInterval.current);
        }
      };
    }
  }, [isGenerating, metrics, validateQuality]);

  /**
   * Update performance metrics
   */
  useEffect(() => {
    if (isGenerating) {
      const monitoringInterval = setInterval(() => {
        monitorPerformance();
      }, 1000);

      return () => clearInterval(monitoringInterval);
    }
  }, [isGenerating, monitorPerformance]);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      if (qualityCheckInterval.current) {
        clearInterval(qualityCheckInterval.current);
      }
    };
  }, []);

  return {
    startGeneration,
    cancelGeneration,
    generationState,
    isGenerating,
    error: error || (contextError ? new Error(contextError) : null),
    progress,
    metrics,
    performance,
    quality: {
      isValid: qualityWarnings.length === 0,
      warnings: qualityWarnings
    }
  };
};

export type { UseGenerationResult };