import React, { useCallback, useEffect, useMemo } from 'react';
import classnames from 'classnames'; // @version ^2.3.2
import debounce from 'lodash/debounce'; // @version ^4.0.8
import { Dropdown, DropdownProps } from '../common/Dropdown';
import { useVideo } from '../../hooks/useVideo';
import { CONTROL_SETTINGS } from '../../config/constants';

// Environment control options based on technical specifications
const WEATHER_OPTIONS = [
  { value: 'clear', label: 'Clear', icon: 'control', state: 'idle' },
  { value: 'rain', label: 'Rain', icon: 'control', state: 'idle' },
  { value: 'snow', label: 'Snow', icon: 'control', state: 'idle' },
  { value: 'fog', label: 'Fog', icon: 'control', state: 'idle' }
] as const;

const LIGHTING_OPTIONS = [
  { value: 'day', label: 'Day', icon: 'control', state: 'idle' },
  { value: 'night', label: 'Night', icon: 'control', state: 'idle' },
  { value: 'sunset', label: 'Sunset', icon: 'control', state: 'idle' },
  { value: 'dawn', label: 'Dawn', icon: 'control', state: 'idle' }
] as const;

const EFFECTS_OPTIONS = [
  { value: 'none', label: 'None', icon: 'control', state: 'idle' },
  { value: 'bloom', label: 'Bloom', icon: 'control', state: 'idle' },
  { value: 'motion_blur', label: 'Motion Blur', icon: 'control', state: 'idle' },
  { value: 'depth_of_field', label: 'Depth of Field', icon: 'control', state: 'idle' }
] as const;

// Props interface with comprehensive control options
export interface EnvironmentControlsProps {
  onWeatherChange?: (weather: string, latency: number) => void;
  onLightingChange?: (lighting: string, latency: number) => void;
  onEffectsChange?: (effects: string, latency: number) => void;
  disabled?: boolean;
  className?: string;
  highContrast?: boolean;
}

export const EnvironmentControls: React.FC<EnvironmentControlsProps> = ({
  onWeatherChange,
  onLightingChange,
  onEffectsChange,
  disabled = false,
  className,
  highContrast = false
}) => {
  // Video state management hook
  const { videoState } = useVideo();

  // Debounced change handlers for performance optimization
  const handleWeatherChange = useMemo(
    () => debounce((value: string) => {
      const startTime = performance.now();
      onWeatherChange?.(value, performance.now() - startTime);
    }, CONTROL_SETTINGS.DEBOUNCE_DELAY),
    [onWeatherChange]
  );

  const handleLightingChange = useMemo(
    () => debounce((value: string) => {
      const startTime = performance.now();
      onLightingChange?.(value, performance.now() - startTime);
    }, CONTROL_SETTINGS.DEBOUNCE_DELAY),
    [onLightingChange]
  );

  const handleEffectsChange = useMemo(
    () => debounce((value: string) => {
      const startTime = performance.now();
      onEffectsChange?.(value, performance.now() - startTime);
    }, CONTROL_SETTINGS.DEBOUNCE_DELAY),
    [onEffectsChange]
  );

  // Cleanup debounced handlers
  useEffect(() => {
    return () => {
      handleWeatherChange.cancel();
      handleLightingChange.cancel();
      handleEffectsChange.cancel();
    };
  }, [handleWeatherChange, handleLightingChange, handleEffectsChange]);

  // Common dropdown props for consistent styling
  const dropdownProps: Partial<DropdownProps> = {
    size: 'md',
    variant: 'game',
    gameState: videoState.status,
    hapticFeedback: true,
    highContrast,
    disabled: disabled || !videoState.isPlaying
  };

  return (
    <div
      className={classnames(
        'gamegen-environment-controls',
        { 'gamegen-environment-controls--high-contrast': highContrast },
        className
      )}
      role="group"
      aria-label="Environment Controls"
    >
      <div className="gamegen-environment-controls__section">
        <Dropdown
          {...dropdownProps}
          options={WEATHER_OPTIONS}
          onChange={handleWeatherChange}
          placeholder="Select Weather"
          aria-label="Weather Control"
        />
      </div>

      <div className="gamegen-environment-controls__section">
        <Dropdown
          {...dropdownProps}
          options={LIGHTING_OPTIONS}
          onChange={handleLightingChange}
          placeholder="Select Lighting"
          aria-label="Lighting Control"
        />
      </div>

      <div className="gamegen-environment-controls__section">
        <Dropdown
          {...dropdownProps}
          options={EFFECTS_OPTIONS}
          onChange={handleEffectsChange}
          placeholder="Select Effects"
          aria-label="Effects Control"
        />
      </div>

      <style jsx>{`
        .gamegen-environment-controls {
          display: flex;
          flex-direction: column;
          gap: var(--spacing-md);
          padding: var(--spacing-lg);
          background-color: rgb(var(--surface-rgb));
          border-radius: var(--border-radius-md);
          box-shadow: var(--shadow-sm);
          transition: all var(--transition-speed-normal) var(--transition-timing);
          contain: content;
        }

        .gamegen-environment-controls--high-contrast {
          background-color: rgb(var(--surface-rgb));
          border: var(--game-border-thickness) solid rgb(var(--border-rgb));
        }

        .gamegen-environment-controls__section {
          display: flex;
          flex-direction: column;
          gap: var(--spacing-sm);
        }

        @media (max-width: 768px) {
          .gamegen-environment-controls {
            padding: var(--spacing-md);
          }
        }
      `}</style>
    </div>
  );
};

export default EnvironmentControls;