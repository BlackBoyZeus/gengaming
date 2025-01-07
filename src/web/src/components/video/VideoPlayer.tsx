import React, { useRef, useEffect, useCallback } from 'react';
import styled from '@emotion/styled'; // ^11.0.0
import { VideoControls } from './VideoControls';
import { useVideo } from '../../hooks/useVideo';
import { VIDEO_SETTINGS, SYSTEM_LIMITS } from '../../config/constants';

// Constants from technical specifications
const TARGET_FPS = SYSTEM_LIMITS.MIN_FRAME_RATE;
const CANVAS_CONTEXT_OPTIONS = {
  alpha: false,
  desynchronized: true,
  willReadFrequently: false
} as const;

// Props interface with comprehensive options
interface VideoPlayerProps {
  width: number;
  height: number;
  className?: string;
  onError?: (error: Error) => void;
  showControls?: boolean;
  highContrast?: boolean;
}

// Styled container with hardware acceleration and performance optimizations
const PlayerContainer = styled.div`
  position: relative;
  width: 100%;
  height: 100%;
  background: rgb(var(--background-rgb));
  border-radius: var(--border-radius-md);
  overflow: hidden;
  contain: content;
  will-change: transform;
  transform: translateZ(0);
  backface-visibility: hidden;

  @media (prefers-reduced-motion: reduce) {
    transition: none;
  }

  @media (forced-colors: active) {
    border: 2px solid CanvasText;
  }
`;

// Hardware-accelerated canvas with performance optimizations
const Canvas = styled.canvas`
  width: 100%;
  height: 100%;
  object-fit: contain;
  will-change: transform;
  transform: translateZ(0);
  image-rendering: pixelated;
  
  @media (prefers-contrast: more) {
    filter: contrast(1.1);
  }
`;

// Enhanced video player component with hardware acceleration
const VideoPlayer: React.FC<VideoPlayerProps> = ({
  width = VIDEO_SETTINGS.DEFAULT_RESOLUTION.width,
  height = VIDEO_SETTINGS.DEFAULT_RESOLUTION.height,
  className,
  onError,
  showControls = true,
  highContrast = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);
  const frameRequestRef = useRef<number>();
  const lastFrameTimeRef = useRef<number>(0);

  const {
    videoState,
    play,
    pause,
    seek,
    metrics
  } = useVideo();

  // Initialize canvas context with hardware acceleration
  const initializeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = width;
    canvas.height = height;

    const context = canvas.getContext('2d', CANVAS_CONTEXT_OPTIONS);
    if (!context) {
      onError?.(new Error('Failed to get canvas context'));
      return;
    }

    // Enable hardware acceleration optimizations
    context.imageSmoothingEnabled = false;
    contextRef.current = context;
  }, [width, height, onError]);

  // Optimized frame rendering with performance monitoring
  const renderFrame = useCallback((timestamp: number) => {
    if (!videoState.isPlaying || !contextRef.current) return;

    const elapsed = timestamp - lastFrameTimeRef.current;
    const frameInterval = 1000 / TARGET_FPS;

    if (elapsed >= frameInterval) {
      const frame = videoState.currentFrame;
      if (frame) {
        try {
          contextRef.current.clearRect(0, 0, width, height);
          contextRef.current.putImageData(frame, 0, 0);
          lastFrameTimeRef.current = timestamp;
        } catch (error) {
          onError?.(error as Error);
        }
      }
    }

    frameRequestRef.current = requestAnimationFrame(renderFrame);
  }, [videoState.isPlaying, videoState.currentFrame, width, height, onError]);

  // Setup canvas and start animation loop
  useEffect(() => {
    initializeCanvas();
    return () => {
      if (frameRequestRef.current) {
        cancelAnimationFrame(frameRequestRef.current);
      }
    };
  }, [initializeCanvas]);

  // Handle playback state changes
  useEffect(() => {
    if (videoState.isPlaying) {
      lastFrameTimeRef.current = performance.now();
      frameRequestRef.current = requestAnimationFrame(renderFrame);
    } else if (frameRequestRef.current) {
      cancelAnimationFrame(frameRequestRef.current);
    }
  }, [videoState.isPlaying, renderFrame]);

  return (
    <PlayerContainer 
      className={className}
      data-high-contrast={highContrast}
      role="region"
      aria-label="Video player"
    >
      <Canvas
        ref={canvasRef}
        role="img"
        aria-label="Video content"
        data-testid="video-canvas"
      />
      {showControls && (
        <VideoControls
          disabled={!videoState.totalFrames}
          onError={onError}
          showFrameRate
        />
      )}
    </PlayerContainer>
  );
};

VideoPlayer.displayName = 'VideoPlayer';

export default VideoPlayer;
export type { VideoPlayerProps };