import React, { memo, useCallback, useEffect, useMemo } from 'react';
import styled from '@emotion/styled'; // ^11.0.0
import debounce from 'lodash/debounce'; // ^4.0.8
import Button from '../common/Button';
import Slider from '../common/Slider';
import { useVideo } from '../../hooks/useVideo';
import { CONTROL_SETTINGS, VIDEO_SETTINGS } from '../../config/constants';

// Enhanced props interface with accessibility and error handling
interface VideoControlsProps {
  className?: string;
  disabled?: boolean;
  onError?: (error: Error) => void;
  showFrameRate?: boolean;
}

// Styled container with enhanced accessibility and mobile support
const ControlsContainer = styled.div`
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  background: rgba(var(--surface-rgb), 0.8);
  border-radius: var(--border-radius-md);
  backdrop-filter: blur(8px);
  contain: content;
  will-change: transform;

  @media (hover: none) {
    padding: var(--spacing-lg);
    gap: var(--spacing-lg);
  }

  @media (prefers-reduced-motion) {
    transition: none;
  }

  @media (forced-colors: active) {
    background: Canvas;
    border: 2px solid CanvasText;
  }
`;

// Enhanced frame information display with high contrast support
const FrameInfo = styled.div`
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  color: rgb(var(--text-rgb));
  font-size: var(--font-size-sm);
  font-family: var(--font-family-monospace);
  user-select: none;

  @media (forced-colors: active) {
    color: CanvasText;
  }

  @media (prefers-contrast: more) {
    font-weight: bold;
  }
`;

// Enhanced video controls component with comprehensive features
const VideoControls = memo(({
  className,
  disabled = false,
  onError,
  showFrameRate = true
}: VideoControlsProps) => {
  const {
    videoState,
    play,
    pause,
    seek,
    metrics
  } = useVideo();

  // Optimized frame rate calculation with performance monitoring
  const frameRate = useMemo(() => {
    return Math.round(metrics?.frame_rate || VIDEO_SETTINGS.DEFAULT_FPS);
  }, [metrics?.frame_rate]);

  // Debounced seek handler for performance
  const handleSeek = useCallback(
    debounce((frame: number) => {
      try {
        seek(frame);
      } catch (error) {
        onError?.(error as Error);
      }
    }, CONTROL_SETTINGS.DEBOUNCE_DELAY),
    [seek, onError]
  );

  // Play/pause handler with error boundary
  const handlePlayPause = useCallback(() => {
    try {
      if (videoState.isPlaying) {
        pause();
      } else {
        play();
      }
    } catch (error) {
      onError?.(error as Error);
    }
  }, [videoState.isPlaying, play, pause, onError]);

  // Cleanup debounced handlers
  useEffect(() => {
    return () => {
      handleSeek.cancel();
    };
  }, [handleSeek]);

  return (
    <ControlsContainer
      className={className}
      role="group"
      aria-label="Video controls"
      data-disabled={disabled}
    >
      <Button
        variant="primary"
        icon={videoState.isPlaying ? 'pause' : 'play'}
        onClick={handlePlayPause}
        disabled={disabled}
        aria-label={videoState.isPlaying ? 'Pause video' : 'Play video'}
        hapticFeedback
        highContrast
      >
        {videoState.isPlaying ? 'Pause' : 'Play'}
      </Button>

      <Slider
        value={videoState.currentFrame}
        min={0}
        max={videoState.totalFrames - 1}
        step={1}
        onChange={handleSeek}
        disabled={disabled}
        label="Frame"
        ariaLabel="Seek video frame"
      />

      <FrameInfo aria-live="polite">
        <span>
          Frame: {videoState.currentFrame + 1}/{videoState.totalFrames}
        </span>
        {showFrameRate && (
          <span aria-label="Frames per second">
            {frameRate} FPS
          </span>
        )}
      </FrameInfo>
    </ControlsContainer>
  );
});

VideoControls.displayName = 'VideoControls';

export default VideoControls;