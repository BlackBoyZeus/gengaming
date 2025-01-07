import React from 'react';
import { render, fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { performance } from 'perf_hooks';

import ControlPanel, { ControlPanelProps } from '../../../src/components/control/ControlPanel';
import { VideoContext, VideoContextType } from '../../../src/contexts/VideoContext';
import { ControlType } from '../../../src/types/api';
import { SYSTEM_LIMITS, CONTROL_SETTINGS } from '../../../src/config/constants';

// Mock performance API
vi.mock('perf_hooks', () => ({
  performance: {
    now: vi.fn(() => Date.now()),
    mark: vi.fn(),
    measure: vi.fn(),
    clearMarks: vi.fn(),
    clearMeasures: vi.fn()
  }
}));

// Enhanced render helper with context and performance monitoring
const renderWithContext = (
  ui: React.ReactElement,
  contextValue: Partial<VideoContextType> = {},
  options: { performanceMonitoring?: boolean } = {}
) => {
  const defaultContext: VideoContextType = {
    videoState: {
      isPlaying: true,
      currentFrame: 0,
      totalFrames: 102,
      frameRate: 24,
      latency: 0,
      bufferHealth: 1,
      error: null,
      status: 'PLAYING',
      metrics: {
        currentFps: 24,
        averageLatency: 45,
        bufferHealth: 1,
        memoryUsage: 0,
        frameDrops: 0
      }
    },
    play: vi.fn(),
    pause: vi.fn(),
    seek: vi.fn(),
    clear: vi.fn(),
    metrics: {},
    ...contextValue
  };

  if (options.performanceMonitoring) {
    performance.now.mockImplementation(() => Date.now());
  }

  return {
    ...render(
      <VideoContext.Provider value={defaultContext}>
        {ui}
      </VideoContext.Provider>
    ),
    mockContext: defaultContext
  };
};

describe('ControlPanel Component', () => {
  // Reset mocks before each test
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Accessibility Tests', () => {
    it('should have proper ARIA attributes and roles', () => {
      const { container } = renderWithContext(<ControlPanel />);
      
      expect(screen.getByRole('region')).toHaveAttribute('aria-label', 'Game Controls');
      expect(screen.getByRole('group', { name: /keyboard controls/i })).toBeInTheDocument();
      expect(screen.getByRole('group', { name: /environment controls/i })).toBeInTheDocument();
    });

    it('should support keyboard navigation', async () => {
      renderWithContext(<ControlPanel />);
      
      const controls = screen.getAllByRole('button');
      await userEvent.tab();
      expect(controls[0]).toHaveFocus();
      
      for (let i = 1; i < controls.length; i++) {
        await userEvent.tab();
        expect(controls[i]).toHaveFocus();
      }
    });

    it('should handle high contrast mode correctly', () => {
      renderWithContext(<ControlPanel highContrast />);
      
      expect(screen.getByRole('region')).toHaveAttribute('data-high-contrast', 'true');
      expect(screen.getByRole('region')).toHaveClass('control-panel--high-contrast');
    });
  });

  describe('Performance Tests', () => {
    it('should maintain control response time within limits', async () => {
      const onControlChange = vi.fn();
      const onLatencyExceeded = vi.fn();
      
      renderWithContext(
        <ControlPanel 
          onControlChange={onControlChange} 
          onLatencyExceeded={onLatencyExceeded}
        />,
        {},
        { performanceMonitoring: true }
      );

      // Simulate rapid keyboard controls
      const startTime = performance.now();
      await userEvent.keyboard('[KeyW]');
      const endTime = performance.now();
      
      expect(endTime - startTime).toBeLessThan(SYSTEM_LIMITS.MAX_CONTROL_LATENCY);
      expect(onLatencyExceeded).not.toHaveBeenCalled();
    });

    it('should debounce control updates correctly', async () => {
      const onControlChange = vi.fn();
      
      renderWithContext(
        <ControlPanel onControlChange={onControlChange} />,
        {},
        { performanceMonitoring: true }
      );

      // Simulate rapid control changes
      for (let i = 0; i < 5; i++) {
        await userEvent.keyboard('[KeyW]');
      }
      
      await waitFor(() => {
        expect(onControlChange).toHaveBeenCalledTimes(1);
      }, { timeout: CONTROL_SETTINGS.DEBOUNCE_DELAY * 2 });
    });
  });

  describe('Error Handling Tests', () => {
    it('should handle and recover from control errors', async () => {
      const onError = vi.fn();
      const { rerender } = renderWithContext(
        <ControlPanel onError={onError} />
      );

      // Simulate error condition
      rerender(
        <ControlPanel 
          onError={onError}
          enabled={false}
        />
      );

      expect(screen.getByRole('region')).toHaveClass('control-panel--disabled');
      expect(onError).not.toHaveBeenCalled();
    });

    it('should display error boundary fallback on critical errors', () => {
      const onError = vi.fn();
      const ThrowError = () => { throw new Error('Test error'); };
      
      renderWithContext(
        <ControlPanel onError={onError}>
          <ThrowError />
        </ControlPanel>
      );

      expect(screen.getByRole('alert')).toHaveTextContent(/control panel error/i);
      expect(onError).toHaveBeenCalled();
    });
  });

  describe('State Management Tests', () => {
    it('should sync control states with video context', async () => {
      const { mockContext } = renderWithContext(<ControlPanel />);

      // Toggle video playback state
      mockContext.videoState.isPlaying = false;
      
      await waitFor(() => {
        expect(screen.getByRole('region')).toHaveAttribute('data-enabled', 'false');
      });
    });

    it('should maintain control state during rapid updates', async () => {
      const onControlChange = vi.fn();
      
      renderWithContext(
        <ControlPanel onControlChange={onControlChange} />
      );

      // Simulate alternating control states
      await userEvent.keyboard('{w>}');
      await userEvent.keyboard('{w/}');
      
      expect(onControlChange).toHaveBeenCalledWith(
        ControlType.KEYBOARD,
        expect.objectContaining({ states: expect.any(Object) })
      );
    });
  });
});