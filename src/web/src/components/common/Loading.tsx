import React from 'react';
import '../../assets/styles/theme.css';

interface LoadingProps {
  size?: 'small' | 'medium' | 'large';
  label?: string;
  className?: string;
  progress?: number;
}

const Loading: React.FC<LoadingProps> = ({
  size = 'medium',
  label = 'Loading...',
  className = '',
  progress,
}) => {
  // Base size dimensions mapped to theme spacing units
  const sizeMap = {
    small: '24px',
    medium: '32px',
    large: '48px',
  };

  // Dynamic styles with performance optimizations
  const styles = {
    container: {
      display: 'inline-flex',
      flexDirection: 'column' as const,
      alignItems: 'center',
      justifyContent: 'center',
      contain: 'layout style', // Optimize rendering performance
    },
    spinner: {
      width: sizeMap[size],
      height: sizeMap[size],
      border: '2px solid rgb(var(--primary-rgb))',
      borderTopColor: 'transparent',
      borderRadius: '50%',
      animation: 'loading-spin var(--transition-speed-normal) linear infinite',
      willChange: 'transform', // Optimize animation performance
      contain: 'layout', // Optimize layout calculations
    },
    label: {
      marginTop: 'var(--spacing-xs)',
      fontSize: size === 'small' ? 'var(--font-size-sm)' : 'var(--font-size-md)',
      color: 'rgb(var(--text-rgb))',
      textAlign: 'center' as const,
    },
    progress: {
      position: 'absolute' as const,
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      fontSize: 'var(--font-size-xs)',
      color: 'rgb(var(--text-rgb))',
    }
  };

  // Keyframe animation definition
  React.useEffect(() => {
    const styleSheet = document.styleSheets[0];
    const keyframes = `
      @keyframes loading-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
      @media (prefers-reduced-motion: reduce) {
        .loading-spinner {
          animation-duration: 1.5s !important;
        }
      }
    `;
    styleSheet.insertRule(keyframes, styleSheet.cssRules.length);
  }, []);

  return (
    <div
      className={`loading-container ${className}`}
      style={styles.container}
      role="status"
      aria-live="polite"
      aria-busy="true"
      aria-label={label}
    >
      <div 
        className="loading-spinner"
        style={styles.spinner}
        aria-hidden="true"
      >
        {progress !== undefined && (
          <div style={styles.progress}>
            {Math.round(progress)}%
          </div>
        )}
      </div>
      <div 
        className="loading-label"
        style={styles.label}
      >
        {label}
      </div>
    </div>
  );
};

export default Loading;