/**
 * GameGen-X Generation Context
 * @version 1.0.0
 * 
 * React context provider for managing video generation state and operations
 * with enhanced performance monitoring and quality metrics tracking.
 */

import { createContext, useContext, useState, useCallback, useEffect } from 'react'; // ^18.0.0
import { GenerationService } from '../services/generation';
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
 * Interface for performance monitoring metrics
 */
interface PerformanceMetrics {
  generationLatency: number;
  controlLatency: number;
  frameRate: number;
  qualityScore: number;
}

/**
 * Interface defining the shape of the generation context
 */
interface GenerationContextType {
  generationState: GenerationState | null;
  isGenerating: boolean;
  error: string | null;
  progress: number;
  metrics: GenerationMetrics;
  performance: PerformanceMetrics;
  startGeneration: (prompt: string, parameters: GenerationParameters) => Promise<string>;
  cancelGeneration: () => Promise<void>;
  validateQuality: () => Promise<boolean>;
  monitorPerformance: () => void;
}

// Create the context with null initial value
const GenerationContext = createContext<GenerationContextType | null>(null);

/**
 * Custom hook for accessing the generation context
 */
export const useGenerationContext = () => {
  const context = useContext(GenerationContext);
  if (!context) {
    throw new Error('useGenerationContext must be used within a GenerationProvider');
  }
  return context;
};

/**
 * Props interface for the GenerationProvider component
 */
interface GenerationProviderProps {
  children: React.ReactNode;
}

/**
 * Generation context provider component
 */
export const GenerationProvider: React.FC<GenerationProviderProps> = ({ children }) => {
  // Initialize service and state
  const generationService = new GenerationService();
  const [generationState, setGenerationState] = useState<GenerationState | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [metrics, setMetrics] = useState<GenerationMetrics>({
    fid_score: 0,
    fvd_score: 0,
    generation_time_ms: 0,
    actual_fps: 0
  });
  const [performance, setPerformance] = useState<PerformanceMetrics>({
    generationLatency: 0,
    controlLatency: 0,
    frameRate: 0,
    qualityScore: 0
  });

  /**
   * Start video generation with enhanced monitoring
   */
  const startGeneration = useCallback(async (
    prompt: string,
    parameters: GenerationParameters
  ): Promise<string> => {
    setIsGenerating(true);
    setError(null);
    setProgress(0);

    try {
      const startTime = Date.now();
      const state = await generationService.startGeneration(prompt, parameters);
      setGenerationState(state);

      // Set up progress monitoring
      generationService.onProgress((updatedState) => {
        setProgress(updatedState.progress);
        setMetrics(updatedState.metrics);
        setGenerationState(updatedState);
      });

      // Update performance metrics
      setPerformance(prev => ({
        ...prev,
        generationLatency: Date.now() - startTime
      }));

      return state.id;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Generation failed';
      setError(errorMessage);
      throw err;
    } finally {
      setIsGenerating(false);
    }
  }, []);

  /**
   * Cancel ongoing generation with cleanup
   */
  const cancelGeneration = useCallback(async (): Promise<void> => {
    try {
      await generationService.cancelGeneration();
      setGenerationState(null);
      setProgress(0);
      setError(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Cancellation failed';
      setError(errorMessage);
    }
  }, []);

  /**
   * Validate generation quality metrics
   */
  const validateQuality = useCallback(async (): Promise<boolean> => {
    if (!generationState) return false;

    const isQualityValid = 
      metrics.fid_score <= MAX_FID_SCORE &&
      metrics.fvd_score <= MAX_FVD_SCORE &&
      metrics.actual_fps >= TARGET_FPS;

    if (!isQualityValid) {
      setError('Generation quality below required thresholds');
    }

    return isQualityValid;
  }, [metrics, generationState]);

  /**
   * Monitor real-time performance metrics
   */
  const monitorPerformance = useCallback(() => {
    if (!generationState) return;

    setPerformance(prev => ({
      ...prev,
      frameRate: metrics.actual_fps,
      qualityScore: Math.max(0, 100 - (metrics.fid_score / MAX_FID_SCORE * 100))
    }));
  }, [metrics, generationState]);

  // Set up automatic performance monitoring
  useEffect(() => {
    if (isGenerating) {
      const interval = setInterval(monitorPerformance, 1000);
      return () => clearInterval(interval);
    }
  }, [isGenerating, monitorPerformance]);

  // Context value
  const contextValue: GenerationContextType = {
    generationState,
    isGenerating,
    error,
    progress,
    metrics,
    performance,
    startGeneration,
    cancelGeneration,
    validateQuality,
    monitorPerformance
  };

  return (
    <GenerationContext.Provider value={contextValue}>
      {children}
    </GenerationContext.Provider>
  );
};

export default GenerationContext;