import React, { memo, useCallback } from 'react';
import '../../assets/styles/theme.css';

interface ProgressBarProps {
  progress: number;
  label?: string;
  className?: string;
  showPercentage?: boolean;
  size?: 'small' | 'medium' | 'large';
  color?: 'primary' | 'secondary' | 'accent';
  ariaLabel?: string;
}

const clampProgress = (value: number): number => {
  return Math.min(Math.max(value, 0), 100);
};

const ProgressBar: React.FC<ProgressBarProps> = memo(({
  progress,
  label,
  className = '',
  showPercentage = true,
  size = 'medium',
  color = 'primary',
  ariaLabel,
}) => {
  const clampedProgress = clampProgress(progress);
  const formattedProgress = `${Math.round(clampedProgress)}%`;

  // Memoize size-based styles
  const getSizeStyles = useCallback(() => {
    switch (size) {
      case 'small':
        return {
          height: 'var(--spacing-xs)',
          fontSize: 'var(--font-size-sm)',
          padding: 'var(--spacing-xs) 0',
        };
      case 'large':
        return {
          height: 'var(--spacing-md)',
          fontSize: 'var(--font-size-md)',
          padding: 'var(--spacing-md) 0',
        };
      default:
        return {
          height: 'var(--spacing-sm)',
          fontSize: 'var(--font-size-sm)',
          padding: 'var(--spacing-sm) 0',
        };
    }
  }, [size]);

  // Memoize color-based styles
  const getColorStyles = useCallback(() => {
    switch (color) {
      case 'secondary':
        return { backgroundColor: 'rgb(var(--secondary-rgb))' };
      case 'accent':
        return { backgroundColor: 'rgb(var(--game-state-active-rgb))' };
      default:
        return { backgroundColor: 'rgb(var(--primary-rgb))' };
    }
  }, [color]);

  const sizeStyles = getSizeStyles();
  const colorStyles = getColorStyles();

  return (
    <div
      className={`progress-bar-container ${className}`}
      role="progressbar"
      aria-valuenow={clampedProgress}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={ariaLabel || label || 'Progress'}
      style={{
        width: '100%',
        ...sizeStyles,
      }}
    >
      {label && (
        <div className="progress-label">
          {label}
        </div>
      )}
      <div 
        className="progress-track"
        style={{
          backgroundColor: 'rgb(var(--background-rgb))',
          borderRadius: 'var(--border-radius-md)',
          overflow: 'hidden',
          position: 'relative',
          height: '100%',
        }}
      >
        <div
          className="progress-fill"
          style={{
            ...colorStyles,
            width: '100%',
            height: '100%',
            transform: `translateX(${clampedProgress - 100}%)`,
            transition: 'transform var(--transition-speed-normal) var(--transition-timing)',
            willChange: 'transform',
          }}
        />
      </div>
      {showPercentage && (
        <div 
          className="progress-percentage"
          aria-hidden="true"
          style={{
            color: 'rgb(var(--text-rgb))',
            marginLeft: 'var(--spacing-xs)',
          }}
        >
          {formattedProgress}
        </div>
      )}

      <style jsx>{`
        .progress-bar-container {
          display: flex;
          align-items: center;
          gap: var(--spacing-xs);
        }

        .progress-label {
          color: rgb(var(--text-rgb));
          font-size: inherit;
          flex-shrink: 0;
        }

        .progress-track {
          flex-grow: 1;
        }

        .progress-percentage {
          flex-shrink: 0;
          font-variant-numeric: tabular-nums;
        }

        @media (prefers-reduced-motion: reduce) {
          .progress-fill {
            transition: none !important;
          }
        }
      `}</style>
    </div>
  );
});

ProgressBar.displayName = 'ProgressBar';

export default ProgressBar;