import React, { useCallback, useEffect } from 'react';
import classnames from 'classnames'; // ^2.3.2
import useKeyboard, { UseKeyboardOptions, KeyboardState } from '../../hooks/useKeyboard';
import { KEY_MAPPINGS, KeyMapping } from '../../utils/keyboard';
import Button from '../common/Button';

export interface KeyboardControlsProps {
  enabled?: boolean;
  onControlChange?: (controlStates: Record<string, boolean>, timestamp: number) => void;
  className?: string;
  debounceMs?: number;
}

const KeyboardControls = React.memo(({
  enabled = true,
  onControlChange,
  className,
  debounceMs = 16
}: KeyboardControlsProps) => {
  // Configure keyboard hook with performance optimizations
  const keyboardOptions: UseKeyboardOptions = {
    enabled,
    debounceMs,
    allowedKeys: Object.keys(KEY_MAPPINGS)
  };

  const {
    controlStates,
    isActive,
    lastKeyPressed,
    responseTime
  } = useKeyboard(keyboardOptions);

  // Handle control state changes with performance timestamp
  useEffect(() => {
    if (onControlChange) {
      onControlChange(controlStates, performance.now());
    }
  }, [controlStates, onControlChange]);

  // Memoized button renderer for performance
  const renderControlButton = useCallback((key: string, action: string) => {
    const isPressed = controlStates[action] || false;
    
    return (
      <Button
        key={key}
        variant="primary"
        size="lg"
        className={classnames(
          'keyboard-control__button',
          { 'keyboard-control__button--active': isPressed }
        )}
        aria-pressed={isPressed}
        aria-label={`${action} control`}
        disabled={!enabled}
        gameState={isPressed ? 'active' : 'idle'}
        hapticFeedback
        style={{
          width: 'var(--game-button-size)',
          height: 'var(--game-button-size)',
          position: 'relative',
          isolation: 'isolate',
          transform: isPressed ? 'scale(0.95)' : 'none',
          transition: 'transform var(--transition-speed-fast) var(--transition-timing)',
          willChange: 'transform',
          contain: 'layout style paint'
        }}
      >
        {key}
      </Button>
    );
  }, [controlStates, enabled]);

  // Container styles with performance optimizations
  const containerStyles: React.CSSProperties = {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: 'var(--spacing-unit)',
    padding: 'var(--spacing-md)',
    backgroundColor: 'rgb(var(--surface-rgb))',
    borderRadius: 'var(--border-radius-md)',
    maxWidth: '320px',
    position: 'relative',
    isolation: 'isolate',
    contain: 'layout style paint',
    willChange: 'transform',
    transform: 'translateZ(0)'
  };

  // Performance monitoring
  useEffect(() => {
    if (responseTime > 50) {
      console.warn(`Keyboard control response time exceeded target: ${responseTime}ms`);
    }
  }, [responseTime]);

  return (
    <div
      className={classnames('keyboard-controls', className)}
      style={containerStyles}
      role="group"
      aria-label="Keyboard controls"
      data-active={isActive}
      data-last-key={lastKeyPressed}
    >
      <div style={{ gridColumn: '2' }}>
        {renderControlButton('W', 'FORWARD')}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', gridColumn: '1 / span 3' }}>
        {renderControlButton('A', 'LEFT')}
        {renderControlButton('S', 'BACKWARD')}
        {renderControlButton('D', 'RIGHT')}
      </div>
      <div style={{ gridColumn: '2' }}>
        {renderControlButton('SPACE', 'ACTION')}
      </div>
    </div>
  );
});

KeyboardControls.displayName = 'KeyboardControls';

export default KeyboardControls;