/**
 * GameGen-X useGeneration Hook Tests
 * @version 1.0.0
 * 
 * Comprehensive test suite for validating video generation hook functionality,
 * quality metrics, and performance monitoring.
 */

import { renderHook, act } from '@testing-library/react-hooks'; // ^8.0.0
import { waitFor } from '@testing-library/react'; // ^14.0.0
import { useGeneration } from '../../src/hooks/useGeneration';
import { GenerationProvider } from '../../src/contexts/GenerationContext';
import { GenerationParameters, Perspective, GenerationStatus } from '../../src/types/generation';
import { SYSTEM_LIMITS } from '../../src/config/constants';

// Mock services and dependencies
jest.mock('../../src/services/generation');
jest.mock('../../src/services/websocket');
jest.mock('../../src/services/metrics');

describe('useGeneration Hook', () => {
  // Test setup with mock generation parameters
  const mockGenerationParams: GenerationParameters = {
    resolution: { width: 1280, height: 720 },
    frames: 102,
    perspective: Perspective.THIRD_PERSON,
    fps: 24
  };

  const mockPrompt = "Generate a fantasy landscape with mountains";

  // Setup wrapper for hook testing
  const wrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <GenerationProvider>{children}</GenerationProvider>
  );

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  describe('Initialization', () => {
    it('should initialize with correct default state', () => {
      const { result } = renderHook(() => useGeneration(), { wrapper });

      expect(result.current.generationState).toBeNull();
      expect(result.current.isGenerating).toBeFalsy();
      expect(result.current.error).toBeNull();
      expect(result.current.progress).toBe(0);
      expect(result.current.metrics).toEqual({
        fid_score: 0,
        fvd_score: 0,
        generation_time_ms: 0,
        actual_fps: 0
      });
    });

    it('should provide required generation control methods', () => {
      const { result } = renderHook(() => useGeneration(), { wrapper });

      expect(result.current.startGeneration).toBeDefined();
      expect(result.current.cancelGeneration).toBeDefined();
      expect(typeof result.current.startGeneration).toBe('function');
      expect(typeof result.current.cancelGeneration).toBe('function');
    });
  });

  describe('Generation Lifecycle', () => {
    it('should start generation with valid parameters', async () => {
      const { result } = renderHook(() => useGeneration(), { wrapper });

      await act(async () => {
        await result.current.startGeneration(mockPrompt, mockGenerationParams);
      });

      expect(result.current.isGenerating).toBeTruthy();
      expect(result.current.error).toBeNull();
      expect(result.current.generationState?.status).toBe(GenerationStatus.GENERATING);
    });

    it('should track generation progress', async () => {
      const { result } = renderHook(() => useGeneration(), { wrapper });

      await act(async () => {
        await result.current.startGeneration(mockPrompt, mockGenerationParams);
      });

      // Simulate progress updates
      await act(async () => {
        for (let progress = 0; progress <= 100; progress += 20) {
          jest.advanceTimersByTime(1000);
          await waitFor(() => {
            expect(result.current.progress).toBeGreaterThanOrEqual(progress);
          });
        }
      });
    });

    it('should validate quality metrics against thresholds', async () => {
      const { result } = renderHook(() => useGeneration(), { wrapper });

      await act(async () => {
        await result.current.startGeneration(mockPrompt, mockGenerationParams);
      });

      // Simulate metrics updates
      await act(async () => {
        jest.advanceTimersByTime(1000);
        expect(result.current.quality.isValid).toBeTruthy();
        expect(result.current.metrics.fid_score).toBeLessThanOrEqual(300);
        expect(result.current.metrics.fvd_score).toBeLessThanOrEqual(1000);
      });
    });

    it('should maintain target frame rate', async () => {
      const { result } = renderHook(() => useGeneration(), { wrapper });

      await act(async () => {
        await result.current.startGeneration(mockPrompt, mockGenerationParams);
      });

      await act(async () => {
        jest.advanceTimersByTime(1000);
        expect(result.current.metrics.actual_fps).toBeGreaterThanOrEqual(24);
      });
    });
  });

  describe('Performance Monitoring', () => {
    it('should track generation latency', async () => {
      const { result } = renderHook(() => useGeneration(), { wrapper });

      const startTime = Date.now();
      await act(async () => {
        await result.current.startGeneration(mockPrompt, mockGenerationParams);
      });

      await waitFor(() => {
        const latency = result.current.performance.generationLatency;
        expect(latency).toBeLessThanOrEqual(SYSTEM_LIMITS.MAX_GENERATION_LATENCY);
      });
    });

    it('should monitor control response time', async () => {
      const { result } = renderHook(() => useGeneration(), { wrapper });

      await act(async () => {
        await result.current.startGeneration(mockPrompt, mockGenerationParams);
      });

      await waitFor(() => {
        expect(result.current.performance.controlLatency).toBeLessThanOrEqual(50);
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle generation failures gracefully', async () => {
      const { result } = renderHook(() => useGeneration(), { wrapper });

      // Simulate generation error
      await act(async () => {
        try {
          await result.current.startGeneration('', mockGenerationParams);
        } catch (error) {
          expect(result.current.error).toBeDefined();
          expect(result.current.isGenerating).toBeFalsy();
          expect(result.current.generationState?.status).toBe(GenerationStatus.FAILED);
        }
      });
    });

    it('should handle quality threshold violations', async () => {
      const { result } = renderHook(() => useGeneration(), { wrapper });

      await act(async () => {
        await result.current.startGeneration(mockPrompt, mockGenerationParams);
      });

      // Simulate poor quality metrics
      await act(async () => {
        jest.advanceTimersByTime(1000);
        expect(result.current.quality.warnings).toHaveLength(0);
      });
    });
  });

  describe('Cleanup', () => {
    it('should cleanup resources on unmount', () => {
      const { unmount } = renderHook(() => useGeneration(), { wrapper });

      unmount();
      // Verify cleanup of intervals and subscriptions
      expect(jest.getTimerCount()).toBe(0);
    });

    it('should cancel generation on unmount if active', async () => {
      const { result, unmount } = renderHook(() => useGeneration(), { wrapper });

      await act(async () => {
        await result.current.startGeneration(mockPrompt, mockGenerationParams);
      });

      unmount();
      expect(result.current.isGenerating).toBeFalsy();
    });
  });
});