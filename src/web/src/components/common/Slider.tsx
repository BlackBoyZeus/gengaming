import React, { useCallback, useEffect, useRef } from 'react';
import styled from '@emotion/styled';
import debounce from 'lodash/debounce';

// @emotion/styled version: ^11.0.0
// react version: ^18.0.0
// lodash version: ^4.0.8

interface SliderProps {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  label?: string;
  ariaLabel?: string;
}

const SliderContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  padding: 8px 0;
  touch-action: none;

  @media (max-width: 768px) {
    padding: 12px 0;
  }
`;

const SliderLabel = styled.label`
  font-size: 14px;
  color: ${props => props.theme.colors.text};
  font-weight: 500;
  user-select: none;

  @media (forced-colors: active) {
    color: CanvasText;
  }
`;

const StyledSlider = styled.input<{ disabled?: boolean }>`
  width: 100%;
  height: 4px;
  background: ${props => props.theme.colors.primary};
  border-radius: 2px;
  outline: none;
  opacity: ${props => props.disabled ? 0.5 : 1};
  transition: all 0.2s ease;
  -webkit-appearance: none;

  &::-webkit-slider-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: ${props => props.theme.colors.primary};
    cursor: ${props => props.disabled ? 'not-allowed' : 'pointer'};
    -webkit-appearance: none;
  }

  &::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: ${props => props.theme.colors.primary};
    cursor: ${props => props.disabled ? 'not-allowed' : 'pointer'};
    border: none;
  }

  &:focus {
    outline: 2px solid ${props => props.theme.colors.focus};
    outline-offset: 2px;
  }

  &:hover:not(:disabled) {
    opacity: 0.9;
  }

  @media (hover: none) {
    touch-action: none;
  }
`;

const ValueDisplay = styled.span`
  font-size: 12px;
  color: ${props => props.theme.colors.text};
  opacity: 0.8;
  transition: opacity 0.2s ease;
  user-select: none;
  margin-left: 8px;

  &:hover {
    opacity: 1;
  }
`;

const Slider: React.FC<SliderProps> = ({
  value,
  min = 24,
  max = 102,
  step = 1,
  onChange,
  disabled = false,
  label,
  ariaLabel,
}) => {
  const sliderRef = useRef<HTMLInputElement>(null);

  // Debounced onChange handler for performance
  const debouncedOnChange = useCallback(
    debounce((newValue: number) => {
      onChange(newValue);
    }, 100),
    [onChange]
  );

  // Clean up debounced handler on unmount
  useEffect(() => {
    return () => {
      debouncedOnChange.cancel();
    };
  }, [debouncedOnChange]);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = Math.min(
      max,
      Math.max(min, Math.round(parseFloat(event.target.value) / step) * step)
    );
    debouncedOnChange(newValue);
  };

  // Keyboard event handler for fine-grained control
  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (disabled) return;

    let newValue = value;
    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowUp':
        newValue = Math.min(max, value + step);
        break;
      case 'ArrowLeft':
      case 'ArrowDown':
        newValue = Math.max(min, value - step);
        break;
      case 'Home':
        newValue = min;
        break;
      case 'End':
        newValue = max;
        break;
      default:
        return;
    }
    event.preventDefault();
    debouncedOnChange(newValue);
  };

  // Touch event handler for mobile devices
  const handleTouchStart = (event: React.TouchEvent<HTMLInputElement>) => {
    if (disabled) return;
    event.stopPropagation();
  };

  const id = `slider-${React.useId()}`;

  return (
    <SliderContainer
      role="group"
      aria-labelledby={label ? id : undefined}
    >
      {label && (
        <SliderLabel id={id}>
          {label}
          <ValueDisplay>{value}</ValueDisplay>
        </SliderLabel>
      )}
      <StyledSlider
        ref={sliderRef}
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onTouchStart={handleTouchStart}
        aria-label={ariaLabel || label}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        aria-disabled={disabled}
        role="slider"
      />
    </SliderContainer>
  );
};

export default Slider;