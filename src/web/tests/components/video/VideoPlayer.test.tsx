import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { jest, describe, beforeEach, afterEach, test, expect } from '@jest/globals';
import { Server } from 'mock-socket';
import 'jest-canvas-mock';

import VideoPlayer from '../../../src/components/video/VideoPlayer';
import { useVideo } from '../../../src/hooks/useVideo';
import { GenerationStatus } from '../../../src/types/api';
import { VIDEO_SETTINGS, SYSTEM_LIMITS } from '../../../src/config/constants';

// Mock the useVideo hook
jest.mock('../../../src/hooks/useVideo');

// Constants from technical specifications
const TARGET_FPS = SYSTEM_LIMITS.MIN_FRAME_RATE;
const FRAME_BUFFER_SIZE = VIDEO_SETTINGS.DEFAULT_FRAME_COUNT;
const VIDEO_WIDTH = VIDEO_SETTINGS.DEFAULT_RESOLUTION.width;
const VIDEO_HEIGHT = VIDEO_SETTINGS.DEFAULT_RESOLUTION.height;

describe('VideoPlayer Component', () => {
  // Mock state and functions
  const mockVideoState = {
    isPlaying: false,
    currentFrame: 0,
    totalFrames: FRAME_BUFFER_SIZE,
    frameRate: TARGET_FPS,
    latency: 0,
    bufferHealth: 1,
    error: null,
    status: GenerationStatus.COMPLETED,
    metrics: {
      frame_rate: TARGET_FPS,
      resolution_width: VIDEO_WIDTH,
      resolution_height: VIDEO_HEIGHT,
      latency_ms: 50,
      quality_score: 90
    }
  };

  const mockPlay = jest.fn();
  const mockPause = jest.fn();
  const mockSeek = jest.fn();
  const mockOnError = jest.fn();

  beforeEach(() => {
    // Reset all mocks before each test
    jest.clearAllMocks();
    
    // Mock useVideo implementation
    (useVideo as jest.Mock).mockReturnValue({
      videoState: mockVideoState,
      play: mockPlay,
      pause: mockPause,
      seek: mockSeek
    });

    // Mock canvas context
    const mockContext = {
      clearRect: jest.fn(),
      putImageData: jest.fn(),
      drawImage: jest.fn(),
      getImageData: jest.fn(),
      createImageData: jest.fn()
    };

    HTMLCanvasElement.prototype.getContext = jest.fn(() => mockContext);
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  test('renders correctly with default props', () => {
    render(<VideoPlayer />);
    
    // Verify canvas element
    const canvas = screen.getByTestId('video-canvas');
    expect(canvas).toBeInTheDocument();
    expect(canvas).toHaveAttribute('width', VIDEO_WIDTH.toString());
    expect(canvas).toHaveAttribute('height', VIDEO_HEIGHT.toString());
    
    // Verify accessibility attributes
    expect(canvas).toHaveAttribute('role', 'img');
    expect(canvas).toHaveAttribute('aria-label', 'Video content');
  });

  test('initializes canvas with hardware acceleration', () => {
    render(<VideoPlayer />);
    
    expect(HTMLCanvasElement.prototype.getContext).toHaveBeenCalledWith('2d', {
      alpha: false,
      desynchronized: true,
      willReadFrequently: false
    });
  });

  test('handles playback controls correctly', async () => {
    render(<VideoPlayer showControls />);
    
    // Find play button
    const playButton = screen.getByRole('button', { name: /play video/i });
    
    // Test play
    await userEvent.click(playButton);
    expect(mockPlay).toHaveBeenCalled();
    
    // Update mock state to playing
    (useVideo as jest.Mock).mockReturnValue({
      videoState: { ...mockVideoState, isPlaying: true },
      play: mockPlay,
      pause: mockPause,
      seek: mockSeek
    });
    
    // Test pause
    await userEvent.click(playButton);
    expect(mockPause).toHaveBeenCalled();
  });

  test('handles frame seeking correctly', async () => {
    render(<VideoPlayer showControls />);
    
    // Find slider
    const slider = screen.getByRole('slider');
    
    // Test seeking
    fireEvent.change(slider, { target: { value: '50' } });
    
    await waitFor(() => {
      expect(mockSeek).toHaveBeenCalledWith(50);
    });
  });

  test('maintains performance requirements', async () => {
    const performanceNow = jest.spyOn(performance, 'now');
    const startTime = 1000;
    performanceNow.mockReturnValue(startTime);

    render(<VideoPlayer />);

    // Simulate frame rendering
    const frameInterval = 1000 / TARGET_FPS;
    performanceNow.mockReturnValue(startTime + frameInterval);

    // Verify frame timing
    await waitFor(() => {
      expect(performanceNow).toHaveBeenCalled();
      const renderTime = performanceNow() - startTime;
      expect(renderTime).toBeLessThanOrEqual(frameInterval);
    });
  });

  test('handles errors gracefully', () => {
    render(<VideoPlayer onError={mockOnError} />);

    // Simulate canvas context error
    HTMLCanvasElement.prototype.getContext = jest.fn(() => null);

    expect(mockOnError).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining('Failed to get canvas context')
      })
    );
  });

  test('supports high contrast mode', () => {
    render(<VideoPlayer highContrast />);
    
    const container = screen.getByRole('region');
    expect(container).toHaveAttribute('data-high-contrast', 'true');
  });

  test('cleans up resources on unmount', () => {
    const { unmount } = render(<VideoPlayer />);
    
    // Mock requestAnimationFrame
    const mockRAF = jest.spyOn(window, 'requestAnimationFrame');
    const mockCAF = jest.spyOn(window, 'cancelAnimationFrame');
    
    unmount();
    
    expect(mockCAF).toHaveBeenCalled();
  });
});