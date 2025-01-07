import React, { useCallback, useEffect, useMemo, useState } from 'react';
import classnames from 'classnames'; // @version ^2.3.2
import { withPerformanceTracking } from '@performance-monitor/react'; // @version ^1.0.0

import KeyboardControls from './KeyboardControls';
import EnvironmentControls from './EnvironmentControls';
import { useVideo } from '../../hooks/useVideo';
import { CONTROL_SETTINGS, SYSTEM_LIMITS } from '../../config/constants';
import { ControlType } from '../../types/api';

// Enhanced props interface with accessibility and performance features
export interface ControlPanelProps {
  enabled?: boolean;
  onControlChange?: (controlType: string, value: any) => void;
  className?: string;
  highContrast?: boolean;
  onError?: (error: Error) => void;
  onLatencyExceeded?: (latency: number) => void;
}

// Error boundary component for control panel
class ControlPanelErrorBoundary extends React.Component<
  { children: React.ReactNode; onError?: (error: Error) => void },
  { hasError: boolean }
> {
  constructor(props: { children: React.ReactNode; onError?: (error: Error) => void }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true };
  }

  componentDidCatch(error: Error): void {
    this.props.onError?.(error);
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      return (
        <div role="alert" className="control-panel__error">
          Control panel error. Please refresh the page.
        </div>
      );
    }
    return this.props.children;
  }
}

// Performance-tracked control panel component
const ControlPanel: React.FC<ControlPanelProps> = ({
  enabled = true,
  onControlChange,
  className,
  highContrast = false,
  onError,
  onLatencyExceeded
}) => {
  // Video state management
  const { videoState } = useVideo();
  const [lastControlTime, setLastControlTime] = useState<number>(0);

  // Debounced control change handler with latency tracking
  const handleControlChange = useCallback((controlType: ControlType, value: any) => {
    const currentTime = performance.now();
    const latency = currentTime - lastControlTime;

    // Track control latency
    if (latency > SYSTEM_LIMITS.MAX_CONTROL_LATENCY) {
      onLatencyExceeded?.(latency);
    }

    setLastControlTime(currentTime);
    onControlChange?.(controlType, value);
  }, [lastControlTime, onControlChange, onLatencyExceeded]);

  // Keyboard control handler with performance optimization
  const handleKeyboardControl = useCallback((controlStates: Record<string, boolean>, timestamp: number) => {
    handleControlChange(ControlType.KEYBOARD, {
      states: controlStates,
      timestamp
    });
  }, [handleControlChange]);

  // Environment control handlers with performance optimization
  const handleEnvironmentControl = useMemo(() => ({
    weather: (value: string, latency: number) => {
      handleControlChange(ControlType.ENVIRONMENT, {
        type: 'weather',
        value,
        latency
      });
    },
    lighting: (value: string, latency: number) => {
      handleControlChange(ControlType.ENVIRONMENT, {
        type: 'lighting',
        value,
        latency
      });
    },
    effects: (value: string, latency: number) => {
      handleControlChange(ControlType.ENVIRONMENT, {
        type: 'effects',
        value,
        latency
      });
    }
  }), [handleControlChange]);

  // Performance monitoring cleanup
  useEffect(() => {
    return () => {
      // Cleanup performance measurements
      performance.clearMarks();
      performance.clearMeasures();
    };
  }, []);

  return (
    <ControlPanelErrorBoundary onError={onError}>
      <div
        className={classnames(
          'control-panel',
          { 'control-panel--disabled': !enabled },
          { 'control-panel--high-contrast': highContrast },
          className
        )}
        role="region"
        aria-label="Game Controls"
        data-enabled={enabled}
        data-high-contrast={highContrast}
      >
        <section className="control-panel__section">
          <h2 className="control-panel__section-title">
            Keyboard Controls
          </h2>
          <KeyboardControls
            enabled={enabled && videoState.isPlaying}
            onControlChange={handleKeyboardControl}
            className="control-panel__keyboard"
            debounceMs={CONTROL_SETTINGS.DEBOUNCE_DELAY}
          />
        </section>

        <section className="control-panel__section">
          <h2 className="control-panel__section-title">
            Environment Controls
          </h2>
          <EnvironmentControls
            onWeatherChange={handleEnvironmentControl.weather}
            onLightingChange={handleEnvironmentControl.lighting}
            onEffectsChange={handleEnvironmentControl.effects}
            disabled={!enabled || !videoState.isPlaying}
            className="control-panel__environment"
            highContrast={highContrast}
          />
        </section>

        <style jsx>{`
          .control-panel {
            display: flex;
            flex-direction: column;
            gap: var(--spacing-lg);
            padding: var(--spacing-xl);
            background-color: rgb(var(--surface-rgb));
            border-radius: var(--border-radius-lg);
            box-shadow: var(--shadow-md);
            transition: opacity var(--transition-speed-normal) var(--transition-timing);
            contain: content;
            will-change: transform;
            transform: translateZ(0);
          }

          .control-panel--disabled {
            opacity: 0.5;
            pointer-events: none;
          }

          .control-panel--high-contrast {
            border: var(--game-border-thickness) solid rgb(var(--border-rgb));
          }

          .control-panel__section {
            display: flex;
            flex-direction: column;
            gap: var(--spacing-md);
          }

          .control-panel__section-title {
            font-size: var(--font-size-lg);
            font-weight: var(--font-weight-bold);
            color: rgb(var(--text-rgb));
            margin-bottom: var(--spacing-sm);
          }

          @media (prefers-reduced-motion: reduce) {
            .control-panel {
              transition: none;
            }
          }

          @media (max-width: 768px) {
            .control-panel {
              padding: var(--spacing-lg);
            }
          }
        `}</style>
      </div>
    </ControlPanelErrorBoundary>
  );
};

// Export performance-tracked component
export default withPerformanceTracking(ControlPanel, {
  id: 'ControlPanel',
  metrics: ['fps', 'latency']
});