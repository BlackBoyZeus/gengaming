import React from 'react';
import { render, fireEvent, waitFor, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { performance } from 'perf_hooks';

import GenerationForm from '../../src/components/generation/GenerationForm';
import { useGeneration } from '../../src/hooks/useGeneration';
import { VIDEO_SETTINGS, SYSTEM_LIMITS } from '../../src/config/constants';

// Mock useGeneration hook
jest.mock('../../src/hooks/useGeneration', () => ({
  useGeneration: jest.fn()
}));

// Mock performance.now for timing tests
jest.spyOn(performance, 'now');

describe('GenerationForm', () => {
  // Default mock implementation
  const mockGenerationHook = {
    startGeneration: jest.fn().mockImplementation(async (params) => ({
      fid: 250,
      fvd: 900,
      fps: 24
    })),
    cancelGeneration: jest.fn(),
    isGenerating: false,
    progress: 0,
    metrics: {
      fid_score: 250,
      fvd_score: 900,
      generation_time_ms: 80,
      actual_fps: 24
    },
    quality: {
      isValid: true,
      warnings: []
    }
  };

  // Mock callback handlers
  const mockHandlers = {
    onGenerationStart: jest.fn(),
    onGenerationComplete: jest.fn(),
    onError: jest.fn()
  };

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    (useGeneration as jest.Mock).mockReturnValue(mockGenerationHook);

    // Set FreeBSD user agent
    Object.defineProperty(window.navigator, 'userAgent', {
      value: 'FreeBSD',
      configurable: true
    });
  });

  test('renders form with all required elements', async () => {
    render(
      <GenerationForm
        onGenerationStart={mockHandlers.onGenerationStart}
        onGenerationComplete={mockHandlers.onGenerationComplete}
        onError={mockHandlers.onError}
      />
    );

    // Verify form structure
    expect(screen.getByRole('form')).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toBeInTheDocument();
    expect(screen.getByLabelText(/video generation prompt/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /generate video/i })).toBeInTheDocument();
  });

  test('validates prompt input requirements', async () => {
    render(
      <GenerationForm
        onGenerationStart={mockHandlers.onGenerationStart}
        onGenerationComplete={mockHandlers.onGenerationComplete}
        onError={mockHandlers.onError}
      />
    );

    const promptInput = screen.getByRole('textbox');
    const submitButton = screen.getByRole('button', { name: /generate video/i });

    // Test minimum length validation
    await userEvent.type(promptInput, 'short');
    expect(submitButton).toBeDisabled();
    expect(screen.getByText(/must be at least/i)).toBeInTheDocument();

    // Test maximum length validation
    const longText = 'a'.repeat(301);
    await userEvent.clear(promptInput);
    await userEvent.type(promptInput, longText);
    expect(submitButton).toBeDisabled();
    expect(screen.getByText(/must not exceed/i)).toBeInTheDocument();

    // Test valid input
    await userEvent.clear(promptInput);
    await userEvent.type(promptInput, 'A valid prompt for video generation');
    expect(submitButton).not.toBeDisabled();
  });

  test('handles generation flow with quality metrics', async () => {
    const startTime = 1000;
    performance.now.mockReturnValue(startTime);

    render(
      <GenerationForm
        onGenerationStart={mockHandlers.onGenerationStart}
        onGenerationComplete={mockHandlers.onGenerationComplete}
        onError={mockHandlers.onError}
      />
    );

    // Submit valid generation request
    const promptInput = screen.getByRole('textbox');
    await userEvent.type(promptInput, 'Generate a video with specific parameters');
    await userEvent.click(screen.getByRole('button', { name: /generate video/i }));

    // Verify generation start
    await waitFor(() => {
      expect(mockGenerationHook.startGeneration).toHaveBeenCalledWith(
        'Generate a video with specific parameters',
        expect.objectContaining({
          resolution: VIDEO_SETTINGS.DEFAULT_RESOLUTION,
          frames: VIDEO_SETTINGS.DEFAULT_FRAME_COUNT,
          fps: VIDEO_SETTINGS.DEFAULT_FPS
        })
      );
    });

    // Verify metrics tracking
    expect(mockHandlers.onGenerationStart).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        fid_score: 0,
        fvd_score: 0,
        generation_time_ms: 0,
        actual_fps: 0
      })
    );

    // Verify completion with final metrics
    await waitFor(() => {
      expect(mockHandlers.onGenerationComplete).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          fid_score: 250,
          fvd_score: 900,
          generation_time_ms: expect.any(Number),
          actual_fps: 24
        })
      );
    });
  });

  test('validates performance requirements', async () => {
    const startTime = performance.now();
    
    render(
      <GenerationForm
        onGenerationStart={mockHandlers.onGenerationStart}
        onGenerationComplete={mockHandlers.onGenerationComplete}
        onError={mockHandlers.onError}
      />
    );

    // Measure form interaction response time
    const promptInput = screen.getByRole('textbox');
    await userEvent.type(promptInput, 'Test prompt');
    
    const interactionTime = performance.now() - startTime;
    expect(interactionTime).toBeLessThan(SYSTEM_LIMITS.MAX_CONTROL_LATENCY);

    // Verify frame rate settings
    expect(mockGenerationHook.metrics.actual_fps).toBeGreaterThanOrEqual(SYSTEM_LIMITS.MIN_FRAME_RATE);
  });

  test('handles FreeBSD platform compatibility', async () => {
    render(
      <GenerationForm
        onGenerationStart={mockHandlers.onGenerationStart}
        onGenerationComplete={mockHandlers.onGenerationComplete}
        onError={mockHandlers.onError}
      />
    );

    // Verify FreeBSD-specific rendering
    const form = screen.getByRole('form');
    expect(form).toHaveClass('generation-form');
    expect(window.navigator.userAgent).toContain('FreeBSD');

    // Test FreeBSD performance
    const startTime = performance.now();
    await userEvent.type(screen.getByRole('textbox'), 'FreeBSD test');
    const responseTime = performance.now() - startTime;
    expect(responseTime).toBeLessThan(SYSTEM_LIMITS.MAX_CONTROL_LATENCY);
  });

  test('handles error states appropriately', async () => {
    // Mock error scenario
    mockGenerationHook.startGeneration.mockRejectedValueOnce(new Error('Generation failed'));

    render(
      <GenerationForm
        onGenerationStart={mockHandlers.onGenerationStart}
        onGenerationComplete={mockHandlers.onGenerationComplete}
        onError={mockHandlers.onError}
      />
    );

    // Trigger error
    await userEvent.type(screen.getByRole('textbox'), 'Error test prompt');
    await userEvent.click(screen.getByRole('button', { name: /generate video/i }));

    // Verify error handling
    await waitFor(() => {
      expect(mockHandlers.onError).toHaveBeenCalledWith(
        expect.objectContaining({
          code: 'GENERATION_FAILED',
          message: 'Generation failed'
        })
      );
    });

    // Verify error display
    expect(screen.getByRole('form')).toHaveClass('generation-form--error');
  });
});