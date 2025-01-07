/**
 * GenerationProgress Component
 * @version 1.0.0
 * 
 * Displays real-time progress of video generation with quality metrics and error handling.
 * Implements performance optimizations to maintain 60fps UI updates.
 */

import React, { memo, useCallback, useRef } from 'react';
import { ErrorBoundary } from 'react-error-boundary'; // ^4.0.0
import ProgressBar from '../common/ProgressBar';
import { useGenerationContext } from '../../contexts/GenerationContext';
import { MAX_FID_SCORE, MAX_FVD_SCORE, TARGET_FPS } from '../../types/generation';

// Props interface for the component
interface GenerationProgressProps {
  className?: string;
}

// Interface for generation quality metrics
interface GenerationMetrics {
  fid: number;
  fvd: number;
  fps: number;
}

/**
 * Custom hook for throttling progress updates to maintain UI performance
 */
const useThrottledProgress = (rawProgress: number): number => {
  const lastUpdateRef = useRef<number>(0);
  const progressRef = useRef<number>(0);

  const now = Date.now();
  if (now - lastUpdateRef.current >= 16) { // ~60fps throttle
    lastUpdateRef.current = now;
    progressRef.current = rawProgress;
  }

  return progressRef.current;
};

/**
 * Formats and color-codes metric values based on thresholds
 */
const formatMetric = (value: number, threshold: number, format = ''): string => {
  const color = value <= threshold ? 'var(--success-rgb)' : 'var(--error-rgb)';
  return `<span style="color: rgb(${color})">${value.toFixed(1)}${format}</span>`;
};

/**
 * Generation progress component with real-time metrics display
 */
const GenerationProgress: React.FC<GenerationProgressProps> = memo(({ className = '' }) => {
  const { progress, generationState, metrics, error } = useGenerationContext();
  const throttledProgress = useThrottledProgress(progress);

  // Memoized error handler
  const handleRetry = useCallback(() => {
    window.location.reload();
  }, []);

  // Error fallback component
  const ErrorFallback = ({ error }: { error: Error }) => (
    <div className="error">
      <span>Component Error: {error.message}</span>
      <button className="retryButton" onClick={handleRetry}>
        Retry
      </button>
    </div>
  );

  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <div className={`generation-progress ${className}`}>
        <ProgressBar
          progress={throttledProgress}
          label="Generating Video"
          color={error ? 'error' : 'primary'}
          size="medium"
          showPercentage
          ariaLabel="Video generation progress"
        />

        {metrics && !error && (
          <div className="metrics">
            <span>
              FID: {formatMetric(metrics.fid_score, MAX_FID_SCORE)}
            </span>
            <span>
              FVD: {formatMetric(metrics.fvd_score, MAX_FVD_SCORE)}
            </span>
            <span>
              FPS: {formatMetric(metrics.actual_fps, TARGET_FPS, ' fps')}
            </span>
          </div>
        )}

        {error && (
          <div className="error">
            <span>{error}</span>
            <button className="retryButton" onClick={handleRetry}>
              Retry Generation
            </button>
          </div>
        )}

        <style jsx>{`
          .generation-progress {
            width: 100%;
            margin-bottom: var(--spacing-md);
            position: relative;
          }

          .metrics {
            display: flex;
            justify-content: space-between;
            margin-top: var(--spacing-sm);
            font-size: var(--font-size-sm);
            color: var(--text-color);
            transition: color 0.3s ease;
            font-family: var(--font-family-monospace);
          }

          .error {
            color: rgb(var(--error-rgb));
            margin-top: var(--spacing-sm);
            font-size: var(--font-size-sm);
            display: flex;
            align-items: center;
            gap: var(--spacing-xs);
          }

          .retryButton {
            padding: var(--spacing-xs) var(--spacing-sm);
            background-color: rgb(var(--primary-rgb));
            color: rgb(var(--surface-rgb));
            border: none;
            border-radius: var(--border-radius-sm);
            cursor: pointer;
            transition: background-color 0.2s ease;
          }

          .retryButton:hover {
            background-color: rgb(var(--primary-rgb), 0.9);
          }

          @media (prefers-reduced-motion: reduce) {
            .metrics,
            .retryButton {
              transition: none;
            }
          }
        `}</style>
      </div>
    </ErrorBoundary>
  );
});

GenerationProgress.displayName = 'GenerationProgress';

export default GenerationProgress;