import React, { useId, useCallback } from 'react'; // ^18.0.0
import styled from '@emotion/styled'; // ^11.0.0
import classnames from 'classnames'; // ^2.3.2

// Interfaces
interface RadioOption {
  value: string;
  label: string;
}

interface RadioGroupProps {
  name: string;
  options: RadioOption[];
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  className?: string;
}

// Styled Components
const RadioGroupContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: var(--spacing-unit);
  padding: var(--spacing-unit);
  position: relative;
  min-width: 200px;

  @media (max-width: 768px) {
    padding: calc(var(--spacing-unit) / 2);
  }

  [dir='rtl'] & {
    text-align: right;
  }
`;

const RadioOptionContainer = styled.label<{ disabled?: boolean }>`
  display: flex;
  align-items: center;
  gap: var(--spacing-unit);
  cursor: ${props => props.disabled ? 'not-allowed' : 'pointer'};
  user-select: none;
  color: var(--text-color);
  font-family: var(--font-family-system);
  font-size: var(--font-size-base);
  transition: all var(--transition-speed) ease-in-out;
  padding: calc(var(--spacing-unit) / 2);
  border-radius: 4px;

  &:hover:not([disabled]) {
    background-color: var(--hover-color);
  }

  &[disabled] {
    opacity: 0.6;
  }

  @media (prefers-reduced-motion) {
    transition: none;
  }
`;

const RadioInput = styled.input`
  appearance: none;
  width: 20px;
  height: 20px;
  border: 2px solid var(--primary-color);
  border-radius: 50%;
  background-color: transparent;
  transition: all var(--transition-speed) ease-in-out;
  position: relative;
  cursor: inherit;

  &:checked {
    background-color: var(--primary-color);
    
    &::after {
      content: '';
      position: absolute;
      width: 10px;
      height: 10px;
      background-color: white;
      border-radius: 50%;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
    }
  }

  &:focus-visible {
    outline: 2px solid var(--focus-color);
    outline-offset: 2px;
  }

  &:disabled {
    opacity: 0.6;
  }

  @media (prefers-reduced-motion) {
    transition: none;
  }
`;

// Component
const RadioGroup: React.FC<RadioGroupProps> = React.memo(({
  name,
  options,
  value,
  onChange,
  disabled = false,
  className
}) => {
  const groupId = useId();
  
  const handleKeyDown = useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;

    const currentIndex = options.findIndex(option => option.value === value);
    let newIndex: number;

    switch (event.key) {
      case 'ArrowDown':
      case 'ArrowRight':
        event.preventDefault();
        newIndex = (currentIndex + 1) % options.length;
        onChange(options[newIndex].value);
        break;
      case 'ArrowUp':
      case 'ArrowLeft':
        event.preventDefault();
        newIndex = currentIndex <= 0 ? options.length - 1 : currentIndex - 1;
        onChange(options[newIndex].value);
        break;
      default:
        break;
    }
  }, [disabled, options, value, onChange]);

  const handleChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return;
    onChange(event.target.value);
  }, [disabled, onChange]);

  return (
    <RadioGroupContainer
      className={classnames('radio-group', className)}
      role="radiogroup"
      aria-labelledby={`${groupId}-label`}
      onKeyDown={handleKeyDown}
    >
      {options.map((option) => {
        const optionId = `${groupId}-${option.value}`;
        return (
          <RadioOptionContainer
            key={option.value}
            htmlFor={optionId}
            disabled={disabled}
          >
            <RadioInput
              type="radio"
              id={optionId}
              name={name}
              value={option.value}
              checked={value === option.value}
              onChange={handleChange}
              disabled={disabled}
              aria-checked={value === option.value}
            />
            {option.label}
          </RadioOptionContainer>
        );
      })}
    </RadioGroupContainer>
  );
});

RadioGroup.displayName = 'RadioGroup';

// Exports
export type { RadioOption, RadioGroupProps };
export default RadioGroup;