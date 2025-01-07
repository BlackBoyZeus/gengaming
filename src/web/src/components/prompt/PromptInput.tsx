import React, { useState, useCallback, useMemo, useRef } from 'react';
import classnames from 'classnames'; // ^2.3.2
import DOMPurify from 'dompurify'; // ^3.0.1
import { useDebounce } from 'use-debounce'; // ^9.0.0

import Input from '../common/Input';
import { useGeneration } from '../../hooks/useGeneration';
import { UI_CONSTANTS } from '../../config/constants';

// Constants for prompt validation and behavior
const MIN_PROMPT_LENGTH = UI_CONSTANTS.MIN_PROMPT_LENGTH;
const MAX_PROMPT_LENGTH = UI_CONSTANTS.MAX_PROMPT_LENGTH;
const VALIDATION_DEBOUNCE_MS = 100;

// Interface for component props
interface PromptInputProps {
  onSubmit: (prompt: string) => Promise<void>;
  onValidationChange?: (isValid: boolean) => void;
  className?: string;
  placeholder?: string;
  disabled?: boolean;
  ariaLabel?: string;
}

// Interface for validation result
interface ValidationResult {
  isValid: boolean;
  errors: string[];
}

/**
 * Validates prompt text against security and quality requirements
 */
const validatePrompt = (text: string): ValidationResult => {
  const errors: string[] = [];
  const sanitizedText = DOMPurify.sanitize(text.trim());

  if (sanitizedText.length < MIN_PROMPT_LENGTH) {
    errors.push(`Prompt must be at least ${MIN_PROMPT_LENGTH} characters`);
  }

  if (sanitizedText.length > MAX_PROMPT_LENGTH) {
    errors.push(`Prompt must not exceed ${MAX_PROMPT_LENGTH} characters`);
  }

  // Check for prohibited patterns
  const prohibitedPatterns = [
    /<script>/i,
    /javascript:/i,
    /data:/i,
    /vbscript:/i,
    /on\w+\s*=/i,
    /style\s*=/i
  ];

  if (prohibitedPatterns.some(pattern => pattern.test(sanitizedText))) {
    errors.push('Prompt contains prohibited content');
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

/**
 * PromptInput Component
 * Specialized input component for handling text prompts in the GameGen-X video generation interface
 */
const PromptInput: React.FC<PromptInputProps> = React.memo(({
  onSubmit,
  onValidationChange,
  className,
  placeholder = 'Enter your video generation prompt...',
  disabled = false,
  ariaLabel = 'Video generation prompt input'
}) => {
  // State management
  const [promptText, setPromptText] = useState('');
  const [validation, setValidation] = useState<ValidationResult>({ isValid: false, errors: [] });
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Generation state from context
  const { isGenerating, generationProgress } = useGeneration();

  // Debounced validation
  const [debouncedText] = useDebounce(promptText, VALIDATION_DEBOUNCE_MS);

  // Memoized validation check
  const validateInput = useCallback((text: string) => {
    const result = validatePrompt(text);
    setValidation(result);
    onValidationChange?.(result.isValid);
    return result;
  }, [onValidationChange]);

  // Effect for debounced validation
  React.useEffect(() => {
    validateInput(debouncedText);
  }, [debouncedText, validateInput]);

  // Handle input changes
  const handleChange = useCallback((event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const sanitizedValue = DOMPurify.sanitize(event.target.value);
    setPromptText(sanitizedValue);
  }, []);

  // Handle form submission
  const handleSubmit = useCallback(async (event: React.FormEvent) => {
    event.preventDefault();
    if (validation.isValid && !isGenerating) {
      try {
        await onSubmit(promptText);
        setPromptText('');
      } catch (error) {
        console.error('Prompt submission failed:', error);
      }
    }
  }, [validation.isValid, isGenerating, onSubmit, promptText]);

  // Memoized class names
  const containerClasses = useMemo(() => classnames(
    'prompt-input-container',
    {
      'is-generating': isGenerating,
      'is-invalid': !validation.isValid,
      'is-disabled': disabled
    },
    className
  ), [isGenerating, validation.isValid, disabled, className]);

  return (
    <form onSubmit={handleSubmit} className={containerClasses}>
      <div className="input-wrapper">
        <textarea
          ref={inputRef}
          value={promptText}
          onChange={handleChange}
          disabled={disabled || isGenerating}
          placeholder={placeholder}
          aria-label={ariaLabel}
          aria-invalid={!validation.isValid}
          aria-describedby="prompt-error"
          className="prompt-textarea"
          rows={3}
        />
        
        {isGenerating && (
          <div 
            className="progress-bar"
            style={{ width: `${generationProgress}%` }}
            role="progressbar"
            aria-valuenow={generationProgress}
            aria-valuemin={0}
            aria-valuemax={100}
          />
        )}
      </div>

      <div className="prompt-footer">
        <div className="character-count" aria-live="polite">
          {promptText.length}/{MAX_PROMPT_LENGTH}
        </div>

        {validation.errors.length > 0 && (
          <div id="prompt-error" className="error-messages" role="alert">
            {validation.errors.map((error, index) => (
              <div key={index} className="error-message">{error}</div>
            ))}
          </div>
        )}

        <button
          type="submit"
          disabled={!validation.isValid || isGenerating || disabled}
          className="submit-button"
          aria-busy={isGenerating}
        >
          {isGenerating ? 'Generating...' : 'Generate Video'}
        </button>
      </div>

      <style jsx>{`
        .prompt-input-container {
          display: flex;
          flex-direction: column;
          gap: var(--spacing-sm);
          width: 100%;
          position: relative;
        }

        .input-wrapper {
          position: relative;
          width: 100%;
        }

        .prompt-textarea {
          width: 100%;
          min-height: 80px;
          resize: vertical;
          padding: var(--spacing-sm);
          font-family: var(--font-family-system);
          font-size: var(--font-size-base);
          border: 2px solid var(--border-color);
          border-radius: var(--border-radius);
          transition: border-color 0.2s ease;
        }

        .prompt-textarea:focus {
          border-color: var(--primary-color);
          outline: none;
        }

        .prompt-footer {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: var(--spacing-sm);
        }

        .character-count {
          font-size: var(--font-size-sm);
          color: var(--text-color-secondary);
        }

        .error-messages {
          color: var(--error-color);
          font-size: var(--font-size-sm);
        }

        .progress-bar {
          position: absolute;
          bottom: 0;
          left: 0;
          height: 2px;
          background-color: var(--primary-color);
          transition: width 0.3s ease;
        }

        .submit-button {
          padding: var(--spacing-sm) var(--spacing-md);
          background-color: var(--primary-color);
          color: var(--text-color-inverse);
          border: none;
          border-radius: var(--border-radius);
          cursor: pointer;
          transition: opacity 0.2s ease;
        }

        .submit-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }
      `}</style>
    </form>
  );
});

PromptInput.displayName = 'PromptInput';

export type { PromptInputProps };
export default PromptInput;