import React, { useMemo } from 'react';
import classnames from 'classnames';
import { ErrorBoundary, useErrorBoundary } from 'react-error-boundary';
import { Loading } from '../common/Loading';
import { Icon } from '../common/Icon';
import { GenerationParameters } from '../../types/generation';

interface PromptPreviewProps {
  promptType: 'canny' | 'motion' | 'pose';
  prompt: string;
  isLoading: boolean;
  parameters: GenerationParameters;
  className?: string;
  onError?: (error: Error) => void;
}

// Error fallback component with accessibility support
const ErrorFallback: React.FC<{ error: Error; resetErrorBoundary: () => void }> = ({
  error,
  resetErrorBoundary
}) => (
  <div 
    role="alert" 
    className={styles.error}
    aria-live="assertive"
  >
    <p>Error loading preview: {error.message}</p>
    <button 
      onClick={resetErrorBoundary}
      className="retry-button"
      aria-label="Retry loading preview"
    >
      Retry
    </button>
  </div>
);

const PromptPreview: React.FC<PromptPreviewProps> = React.memo(({
  promptType,
  prompt,
  isLoading,
  parameters,
  className,
  onError
}) => {
  const { showBoundary } = useErrorBoundary();

  // Memoize icon selection based on prompt type
  const iconName = useMemo(() => {
    switch (promptType) {
      case 'canny':
        return 'control';
      case 'motion':
        return 'generate';
      case 'pose':
        return 'control';
      default:
        return 'generate';
    }
  }, [promptType]);

  // Memoize preview container class names
  const containerClasses = useMemo(() => 
    classnames(
      styles.preview,
      {
        [styles.cannyPreview]: promptType === 'canny',
        [styles.motionPreview]: promptType === 'motion',
        [styles.posePreview]: promptType === 'pose'
      },
      className
    ),
    [promptType, className]
  );

  // Handle errors during rendering
  const handleError = (error: Error) => {
    console.error('PromptPreview error:', error);
    onError?.(error);
    showBoundary(error);
  };

  return (
    <ErrorBoundary FallbackComponent={ErrorFallback} onError={handleError}>
      <div 
        className={containerClasses}
        role="region"
        aria-label={`${promptType} prompt preview`}
        tabIndex={0}
      >
        <div className={styles.header}>
          <Icon 
            name={iconName}
            size="md"
            ariaLabel={`${promptType} type indicator`}
          />
          <span className="sr-only">{promptType} preview</span>
        </div>

        <div 
          className={styles.content}
          aria-busy={isLoading}
        >
          {isLoading ? (
            <Loading 
              size="medium"
              label="Loading preview..."
            />
          ) : (
            <div 
              className="preview-content"
              aria-live="polite"
            >
              <p className="preview-prompt">
                {prompt}
              </p>
            </div>
          )}
        </div>

        <div className={styles.parameters}>
          <dl>
            <dt>Resolution:</dt>
            <dd>{parameters.resolution.width}x{parameters.resolution.height}</dd>
            <dt>Frames:</dt>
            <dd>{parameters.frames}</dd>
            <dt>Perspective:</dt>
            <dd>{parameters.perspective}</dd>
            <dt>FPS:</dt>
            <dd>{parameters.fps}</dd>
          </dl>
        </div>
      </div>
    </ErrorBoundary>
  );
});

PromptPreview.displayName = 'PromptPreview';

// CSS Module styles
const styles = {
  preview: {
    padding: '1rem',
    borderRadius: '0.5rem',
    background: 'var(--background-secondary)',
    minHeight: '200px',
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
    position: 'relative',
    outline: 'none',
    transition: 'all 0.2s ease-in-out',
    '@media (prefers-reduced-motion)': {
      transition: 'none'
    }
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    color: 'var(--text-primary)',
    '@media (prefers-contrast: more)': {
      color: 'var(--high-contrast-text)'
    }
  },
  content: {
    flex: '1',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative'
  },
  parameters: {
    fontSize: '0.875rem',
    color: 'var(--text-secondary)',
    '@media (prefers-contrast: more)': {
      color: 'var(--high-contrast-text)'
    }
  },
  error: {
    color: 'var(--error)',
    padding: '1rem',
    border: '1px solid var(--error)',
    borderRadius: '0.25rem',
    marginTop: '0.5rem'
  }
} as const;

export default PromptPreview;