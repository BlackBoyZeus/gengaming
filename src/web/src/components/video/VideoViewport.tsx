import React, { memo, useCallback, useEffect, useRef } from 'react';
import styled from '@emotion/styled'; // ^11.0.0
import { ErrorBoundary } from 'react-error-boundary'; // ^4.0.0
import VideoPlayer from './VideoPlayer';
import VideoControls from './VideoControls';
import { useVideo } from '../../hooks/useVideo';
import { VIDEO_SETTINGS, SYSTEM_LIMITS } from '../../config/constants';

// Constants from technical specifications
const TARGET_FPS = SYSTEM_LIMITS.MIN_FRAME_RATE;
const DEFAULT_WIDTH = VIDEO_SETTINGS.DEFAULT_RESOLUTION.width;
const DEFAULT_HEIGHT = VIDEO_SETTINGS.DEFAULT_RESOLUTION.height;
const PERFORMANCE_BUDGET_MS = 1000 / TARGET_FPS;

// Hardware-accelerated container for video viewport
const ViewportContainer = styled.div`
  position: relative;
  width: 100%;
  height: 100%;
  background: rgb(var(--background-rgb));
  border-radius: var(--border-radius-md);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transform: translateZ(0);
  will-change: transform;
  contain: layout size;

  @media (max-width: 768px) {
    border-radius: 0;
  }

  @media (forced-colors: active) {
    border: 2px solid CanvasText;
  }
`;

// Performance-optimized wrapper for video player
const PlayerWrapper = styled.div`
  flex: 1;
  position: relative;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  contain: layout style;
  touch-action: none;
`;

// Accessible wrapper for video controls
const ControlsWrapper = styled.div`
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: var(--spacing-md);
  z-index: 10;
  background: linear-gradient(transparent, rgba(0,0,0,0.7));

  @media (hover: none) {
    opacity: 1;
  }

  @media (hover: hover) {
    opacity: 0;
    transition: opacity 0.2s;
    &:hover {
      opacity: 1;
    }
  }
`;

// Error fallback component
const ErrorFallback = styled.div`
  padding: var(--spacing-lg);
  color: rgb(var(--error-rgb));
  text-align: center;
`;

interface VideoViewportProps {
  width?: number;
  height?: number;
  className?: string;
  onError?: (error: Error) => void;
}

const VideoViewport = memo(({
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
  className,
  onError
}: VideoViewportProps) => {
  const frameTimeRef = useRef<number>(0);
  const performanceObserverRef = useRef<PerformanceObserver>();

  const {
    videoState,
    play,
    pause,
    seek,
    metrics
  } = useVideo();

  // Monitor frame performance
  useEffect(() => {
    if (!window.PerformanceObserver) return;

    performanceObserverRef.current = new PerformanceObserver((entries) => {
      entries.getEntries().forEach((entry) => {
        if (entry.duration > PERFORMANCE_BUDGET_MS) {
          console.warn(`Frame exceeded budget: ${entry.duration.toFixed(2)}ms`);
        }
      });
    });

    performanceObserverRef.current.observe({ entryTypes: ['frame'] });

    return () => performanceObserverRef.current?.disconnect();
  }, []);

  // Handle video errors
  const handleError = useCallback((error: Error) => {
    console.error('Video viewport error:', error);
    onError?.(error);
  }, [onError]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback((event: React.KeyboardEvent) => {
    if (event.key === 'Space') {
      event.preventDefault();
      videoState.isPlaying ? pause() : play();
    }
  }, [videoState.isPlaying, play, pause]);

  return (
    <ErrorBoundary
      FallbackComponent={({ error }) => (
        <ErrorFallback role="alert">
          {error.message}
        </ErrorFallback>
      )}
      onError={handleError}
    >
      <ViewportContainer
        className={className}
        role="region"
        aria-label="Video viewport"
        onKeyDown={handleKeyDown}
        tabIndex={0}
      >
        <PlayerWrapper>
          <VideoPlayer
            width={width}
            height={height}
            onError={handleError}
          />
        </PlayerWrapper>

        <ControlsWrapper>
          <VideoControls
            disabled={!videoState.totalFrames}
            onError={handleError}
            showFrameRate
          />
        </ControlsWrapper>
      </ViewportContainer>
    </ErrorBoundary>
  );
});

VideoViewport.displayName = 'VideoViewport';

export default VideoViewport;
export type { VideoViewportProps };