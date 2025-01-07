import React, { memo, useCallback, useRef, useEffect } from 'react';
import classnames from 'classnames'; // @version ^2.3.2
import { Loading } from './Loading';
import { Icon } from './Icon';

// Types for button variants and sizes
type ButtonVariant = 'primary' | 'secondary' | 'outline' | 'text';
type ButtonSize = 'sm' | 'md' | 'lg';
type IconPosition = 'left' | 'right';

// Game state type for conditional styling
type GameState = 'idle' | 'loading' | 'active' | 'error';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: string;
  iconPosition?: IconPosition;
  loading?: boolean;
  disabled?: boolean;
  fullWidth?: boolean;
  hapticFeedback?: boolean;
  gameState?: GameState;
  highContrast?: boolean;
  children: React.ReactNode;
  className?: string;
}

const Button = memo(({
  variant = 'primary',
  size = 'md',
  icon,
  iconPosition = 'left',
  loading = false,
  disabled = false,
  fullWidth = false,
  hapticFeedback = true,
  gameState = 'idle',
  highContrast = false,
  children,
  className,
  onClick,
  ...props
}: ButtonProps) => {
  const buttonRef = useRef<HTMLButtonElement>(null);
  const isRTL = document.dir === 'rtl';

  // Setup haptic feedback
  useEffect(() => {
    if (!hapticFeedback || !window.navigator.vibrate) return;
    
    const button = buttonRef.current;
    if (!button) return;

    const handleHaptic = () => {
      if (!disabled && !loading) {
        window.navigator.vibrate(50);
      }
    };

    button.addEventListener('touchstart', handleHaptic);
    return () => button.removeEventListener('touchstart', handleHaptic);
  }, [hapticFeedback, disabled, loading]);

  // Debounced click handler
  const handleClick = useCallback((event: React.MouseEvent<HTMLButtonElement>) => {
    if (loading || disabled) return;
    onClick?.(event);
  }, [onClick, loading, disabled]);

  // Compute class names with performance optimizations
  const buttonClasses = classnames(
    'gamegen-button',
    `gamegen-button--${variant}`,
    `gamegen-button--${size}`,
    `gamegen-button--${gameState}`,
    {
      'gamegen-button--loading': loading,
      'gamegen-button--disabled': disabled,
      'gamegen-button--full-width': fullWidth,
      'gamegen-button--high-contrast': highContrast,
      'gamegen-button--rtl': isRTL,
    },
    className
  );

  // Base styles with hardware acceleration and containment
  const styles: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 'var(--spacing-unit)',
    padding: `var(--spacing-${size === 'sm' ? 'xs' : size === 'md' ? 'sm' : 'md'})`,
    fontSize: `var(--font-size-${size})`,
    borderRadius: 'var(--border-radius-md)',
    transition: 'transform var(--transition-speed-fast) var(--transition-timing)',
    cursor: disabled || loading ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.6 : 1,
    width: fullWidth ? '100%' : 'auto',
    willChange: 'transform, opacity',
    contain: 'layout style paint',
    transform: 'translateZ(0)',
    backfaceVisibility: 'hidden',
    WebkitTapHighlightColor: 'transparent',
  };

  return (
    <button
      ref={buttonRef}
      className={buttonClasses}
      style={styles}
      disabled={disabled || loading}
      onClick={handleClick}
      aria-busy={loading}
      aria-disabled={disabled}
      data-game-state={gameState}
      data-high-contrast={highContrast}
      {...props}
    >
      {loading && (
        <Loading 
          size="small"
          className="gamegen-button__loader"
          aria-hidden="true"
        />
      )}
      
      {icon && iconPosition === (isRTL ? 'right' : 'left') && !loading && (
        <Icon
          name={icon as any}
          size={size === 'sm' ? 'sm' : size === 'md' ? 'md' : 'lg'}
          className="gamegen-button__icon"
          ariaLabel=""
          aria-hidden="true"
        />
      )}
      
      <span className="gamegen-button__content">
        {children}
      </span>

      {icon && iconPosition === (isRTL ? 'left' : 'right') && !loading && (
        <Icon
          name={icon as any}
          size={size === 'sm' ? 'sm' : size === 'md' ? 'md' : 'lg'}
          className="gamegen-button__icon"
          ariaLabel=""
          aria-hidden="true"
        />
      )}
    </button>
  );
});

Button.displayName = 'Button';

export default Button;