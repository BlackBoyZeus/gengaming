import React, { useCallback, useEffect, useState, memo } from 'react';
import styled from '@emotion/styled';
import { ErrorBoundary } from 'react-error-boundary';

import DashboardLayout from '../layouts/DashboardLayout';
import GenerationForm from '../components/generation/GenerationForm';
import VideoViewport from '../components/video/VideoViewport';
import { useGeneration } from '../hooks/useGeneration';
import { GenerationMetrics } from '../types/generation';
import { VIDEO_SETTINGS, UI_CONSTANTS } from '../config/constants';

// Styled components with hardware acceleration and performance optimizations
const PageContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
  max-width: 1280px;
  margin: 0 auto;
  padding: var(--spacing-lg);
  width: 100%;
  transform: translateZ(0);
  will-change: transform;
  contain: layout style paint;

  @media (max-width: 768px) {
    padding: var(--spacing-md);
  }
`;

const ViewportContainer = styled.div`
  width: 100%;
  aspect-ratio: 16/9;
  background: rgb(var(--background-rgb));
  border-radius: var(--border-radius);
  overflow: hidden;
  position: relative;
  box-shadow: var(--shadow-md);
  transform: translateZ(0);
  will-change: transform;
  contain: layout size;

  @media (max-width: 768px) {
    border-radius: var(--border-radius-sm);
  }
`;

// Error fallback component
const ErrorFallback = styled.div`
  padding: var(--spacing-lg);
  color: rgb(var(--error-rgb));
  background: rgb(var(--surface-rgb));
  border-radius: var(--border-radius);
  text-align: center;
`;

// Interface for generation errors
interface GenerationError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

const Generate: React.FC = memo(() => {
  // Access generation context
  const { generationState, error: generationError } = useGeneration();
  const [error, setError] = useState<GenerationError | null>(null);

  // Handle generation start with metrics tracking
  const handleGenerationStart = useCallback((id: string, metrics: GenerationMetrics) => {
    console.debug('Generation started:', { id, metrics });
    setError(null);
  }, []);

  // Handle generation completion with quality validation
  const handleGenerationComplete = useCallback((id: string, metrics: GenerationMetrics) => {
    console.debug('Generation completed:', { id, metrics });
    
    // Validate generation quality
    if (metrics.fid_score > 300 || metrics.fvd_score > 1000) {
      setError({
        code: 'QUALITY_ERROR',
        message: 'Generated video does not meet quality requirements',
        details: { metrics }
      });
    }
  }, []);

  // Handle generation errors
  const handleError = useCallback((error: GenerationError) => {
    console.error('Generation error:', error);
    setError(error);
  }, []);

  // Clean up error state on unmount
  useEffect(() => {
    return () => {
      setError(null);
    };
  }, []);

  return (
    <ErrorBoundary
      FallbackComponent={({ error }) => (
        <ErrorFallback role="alert">
          <h3>Application Error</h3>
          <p>{error.message}</p>
        </ErrorFallback>
      )}
      onError={console.error}
    >
      <DashboardLayout>
        <PageContainer>
          <ViewportContainer>
            <VideoViewport
              width={VIDEO_SETTINGS.DEFAULT_RESOLUTION.width}
              height={VIDEO_SETTINGS.DEFAULT_RESOLUTION.height}
              onError={handleError}
            />
          </ViewportContainer>

          <GenerationForm
            onGenerationStart={handleGenerationStart}
            onGenerationComplete={handleGenerationComplete}
            onError={handleError}
            className="generation-form"
          />

          {error && (
            <div
              role="alert"
              aria-live="assertive"
              className="error-message"
              style={{
                color: 'rgb(var(--error-rgb))',
                padding: 'var(--spacing-md)',
                borderRadius: 'var(--border-radius)',
                backgroundColor: 'rgb(var(--error-rgb), 0.1)'
              }}
            >
              {error.message}
            </div>
          )}
        </PageContainer>

        <style jsx>{`
          .generation-form {
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
          }

          .error-message {
            animation: fadeIn 0.3s ease;
          }

          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
          }

          @media (prefers-reduced-motion: reduce) {
            .error-message {
              animation: none;
            }
          }
        `}</style>
      </DashboardLayout>
    </ErrorBoundary>
  );
});

Generate.displayName = 'Generate';

export default Generate;