import React, { useState, useCallback, useMemo } from 'react';
import classnames from 'classnames'; // ^2.3.2
import PromptInput from '../prompt/PromptInput';
import GenerationSettings from './GenerationSettings';
import GenerationProgress from './GenerationProgress';
import { useGeneration } from '../../hooks/useGeneration';
import { GenerationMetrics, GenerationParameters } from '../../types/generation';
import { UI_CONSTANTS, SYSTEM_LIMITS } from '../../config/constants';

// Interface for component props
export interface GenerationFormProps {
  onGenerationStart: (id: string, metrics: GenerationMetrics) => void;
  onGenerationComplete: (id: string, metrics: GenerationMetrics) => void;
  onError: (error: GenerationError) => void;
  className?: string;
  highContrastMode?: boolean;
}

// Interface for generation errors
interface GenerationError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

// Interface for form validation state
interface ValidationState {
  prompt: boolean;
  settings: boolean;
  quality: boolean;
}

/**
 * GenerationForm Component
 * Main form interface for video generation with enhanced validation and performance tracking
 */
const GenerationForm: React.FC<GenerationFormProps> = React.memo(({
  onGenerationStart,
  onGenerationComplete,
  onError,
  className,
  highContrastMode = false
}) => {
  // Access generation context
  const {
    startGeneration,
    cancelGeneration,
    isGenerating,
    generationState,
    error: generationError,
    progress,
    metrics,
    quality
  } = useGeneration();

  // Form validation state
  const [validation, setValidation] = useState<ValidationState>({
    prompt: false,
    settings: false,
    quality: true
  });

  // Generation parameters state
  const [parameters, setParameters] = useState<GenerationParameters | null>(null);

  // Memoized class names
  const formClasses = useMemo(() => classnames(
    'generation-form',
    {
      'generation-form--generating': isGenerating,
      'generation-form--error': !!generationError,
      'generation-form--high-contrast': highContrastMode
    },
    className
  ), [isGenerating, generationError, highContrastMode, className]);

  // Handle prompt validation changes
  const handlePromptValidation = useCallback((isValid: boolean) => {
    setValidation(prev => ({ ...prev, prompt: isValid }));
  }, []);

  // Handle settings validation changes
  const handleSettingsValidation = useCallback((params: GenerationParameters, isValid: boolean) => {
    setParameters(params);
    setValidation(prev => ({ ...prev, settings: isValid }));
  }, []);

  // Handle form submission
  const handleSubmit = useCallback(async (prompt: string) => {
    if (!parameters || !validation.prompt || !validation.settings) {
      return;
    }

    try {
      const startTime = performance.now();
      const generationId = await startGeneration(prompt, parameters);

      // Track initial metrics
      const initialMetrics: GenerationMetrics = {
        fid_score: 0,
        fvd_score: 0,
        generation_time_ms: 0,
        actual_fps: 0
      };

      onGenerationStart(generationId, initialMetrics);

      // Monitor generation progress
      if (generationState) {
        const finalMetrics = {
          ...generationState.metrics,
          generation_time_ms: performance.now() - startTime
        };
        onGenerationComplete(generationId, finalMetrics);
      }
    } catch (error) {
      const enhancedError: GenerationError = {
        code: 'GENERATION_FAILED',
        message: error instanceof Error ? error.message : 'Generation failed',
        details: { prompt, parameters }
      };
      onError(enhancedError);
    }
  }, [parameters, validation, startGeneration, generationState, onGenerationStart, onGenerationComplete, onError]);

  return (
    <div 
      className={formClasses}
      role="form"
      aria-label="Video Generation Form"
    >
      <PromptInput
        onSubmit={handleSubmit}
        onValidationChange={handlePromptValidation}
        disabled={isGenerating}
        className="generation-form__prompt"
      />

      <GenerationSettings
        onSettingsChange={handleSettingsValidation}
        disabled={isGenerating}
        className="generation-form__settings"
        onValidationError={(error) => {
          onError({
            code: 'SETTINGS_VALIDATION_ERROR',
            message: error.message,
            details: { type: error.type }
          });
        }}
      />

      {(isGenerating || progress > 0) && (
        <GenerationProgress
          className="generation-form__progress"
        />
      )}

      <style jsx>{`
        .generation-form {
          display: flex;
          flex-direction: column;
          gap: var(--spacing-md);
          width: 100%;
          max-width: 800px;
          margin: 0 auto;
          padding: var(--spacing-md);
          background-color: rgb(var(--surface-rgb));
          border-radius: var(--border-radius-lg);
          box-shadow: var(--shadow-md);
          transition: all var(--transition-speed-normal) var(--transition-timing);
        }

        .generation-form--generating {
          opacity: 0.8;
          pointer-events: none;
        }

        .generation-form--error {
          border: var(--game-border-thickness) solid rgb(var(--error-rgb));
        }

        .generation-form--high-contrast {
          border: var(--game-border-thickness) solid rgb(var(--text-rgb));
          box-shadow: none;
        }

        .generation-form__prompt,
        .generation-form__settings,
        .generation-form__progress {
          width: 100%;
        }

        @media (prefers-reduced-motion: reduce) {
          .generation-form {
            transition: none;
          }
        }
      `}</style>
    </div>
  );
});

GenerationForm.displayName = 'GenerationForm';

export type { GenerationFormProps, GenerationError };
export default GenerationForm;