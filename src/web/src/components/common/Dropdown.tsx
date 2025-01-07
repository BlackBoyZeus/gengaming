import React, { useCallback, useRef, useState, useEffect } from 'react';
import classnames from 'classnames'; // @version ^2.3.2
import { Icon } from './Icon';
import { Button } from './Button';

// Type definitions for game-specific states and variants
type GameState = 'idle' | 'loading' | 'active' | 'error';
type IconName = 'control' | 'generate';

// Interface for dropdown options with gaming context
export interface DropdownOption {
  value: string | number;
  label: string;
  disabled?: boolean;
  icon?: IconName;
  tooltip?: string;
  state?: GameState;
}

// Props interface with gaming-specific features
export interface DropdownProps {
  options: DropdownOption[];
  value?: string | number;
  onChange: (value: string | number) => void;
  placeholder?: string;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'primary' | 'secondary' | 'outline' | 'game';
  gameState?: GameState;
  hapticFeedback?: boolean;
  highContrast?: boolean;
  disabled?: boolean;
  className?: string;
  'aria-label'?: string;
}

// Custom hook for optimized dropdown state management
const useDropdownState = (props: DropdownProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [focusedIndex, setFocusedIndex] = useState(-1);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Handle haptic feedback
  const triggerHaptic = useCallback(() => {
    if (props.hapticFeedback && window.navigator.vibrate) {
      window.navigator.vibrate(50);
    }
  }, [props.hapticFeedback]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return {
    isOpen,
    setIsOpen,
    focusedIndex,
    setFocusedIndex,
    dropdownRef,
    triggerHaptic
  };
};

export const Dropdown: React.FC<DropdownProps> = ({
  options,
  value,
  onChange,
  placeholder = 'Select option',
  size = 'md',
  variant = 'primary',
  gameState = 'idle',
  hapticFeedback = true,
  highContrast = false,
  disabled = false,
  className,
  'aria-label': ariaLabel,
}) => {
  const {
    isOpen,
    setIsOpen,
    focusedIndex,
    setFocusedIndex,
    dropdownRef,
    triggerHaptic
  } = useDropdownState({ options, hapticFeedback });

  // Find selected option
  const selectedOption = options.find(opt => opt.value === value);

  // Handle option selection with gaming optimizations
  const handleSelect = useCallback((option: DropdownOption) => {
    if (option.disabled) return;
    
    triggerHaptic();
    onChange(option.value);
    setIsOpen(false);
  }, [onChange, triggerHaptic]);

  // Keyboard navigation handler
  const handleKeyDown = useCallback((event: React.KeyboardEvent) => {
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        setFocusedIndex(prev => Math.min(prev + 1, options.length - 1));
        break;
      case 'ArrowUp':
        event.preventDefault();
        setFocusedIndex(prev => Math.max(prev - 1, 0));
        break;
      case 'Enter':
      case ' ':
        event.preventDefault();
        if (focusedIndex >= 0) {
          handleSelect(options[focusedIndex]);
        }
        break;
      case 'Escape':
        setIsOpen(false);
        break;
    }
  }, [focusedIndex, handleSelect, options]);

  // Dynamic styles with hardware acceleration
  const styles: Record<string, React.CSSProperties> = {
    base: {
      position: 'relative',
      display: 'inline-block',
      fontFamily: 'var(--font-family-system)',
      width: 'auto',
      willChange: 'transform',
      transform: 'translate3d(0,0,0)',
      containIntrinsicSize: 'auto'
    },
    trigger: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: 'var(--spacing-unit)',
      width: '100%',
      textAlign: 'left',
      transition: 'transform 100ms ease-out'
    },
    options: {
      position: 'absolute',
      top: '100%',
      left: '0',
      right: '0',
      marginTop: 'var(--spacing-xs)',
      backgroundColor: 'rgb(var(--surface-rgb))',
      borderRadius: 'var(--border-radius-md)',
      boxShadow: 'var(--shadow-md)',
      zIndex: 10,
      maxHeight: '300px',
      overflowY: 'auto',
      transform: 'translate3d(0,0,0)',
      backfaceVisibility: 'hidden'
    }
  };

  return (
    <div
      ref={dropdownRef}
      className={classnames(
        'gamegen-dropdown',
        `gamegen-dropdown--${size}`,
        `gamegen-dropdown--${variant}`,
        `gamegen-dropdown--${gameState}`,
        { 'gamegen-dropdown--high-contrast': highContrast },
        className
      )}
      style={styles.base}
      onKeyDown={handleKeyDown}
      role="combobox"
      aria-expanded={isOpen}
      aria-haspopup="listbox"
      aria-label={ariaLabel}
      data-game-state={gameState}
      data-high-contrast={highContrast}
    >
      <Button
        variant={variant}
        size={size}
        disabled={disabled}
        gameState={gameState}
        hapticFeedback={hapticFeedback}
        highContrast={highContrast}
        onClick={() => !disabled && setIsOpen(!isOpen)}
        aria-controls="dropdown-options"
        style={styles.trigger}
      >
        <span className="gamegen-dropdown__label">
          {selectedOption ? selectedOption.label : placeholder}
        </span>
        <Icon
          name="control"
          size={size === 'sm' ? 'sm' : 'md'}
          className={classnames('gamegen-dropdown__icon', {
            'gamegen-dropdown__icon--open': isOpen
          })}
          ariaLabel=""
        />
      </Button>

      {isOpen && (
        <ul
          id="dropdown-options"
          className="gamegen-dropdown__options"
          role="listbox"
          aria-label={`${ariaLabel} options`}
          style={styles.options}
        >
          {options.map((option, index) => (
            <li
              key={option.value}
              className={classnames(
                'gamegen-dropdown__option',
                {
                  'gamegen-dropdown__option--focused': index === focusedIndex,
                  'gamegen-dropdown__option--selected': option.value === value,
                  'gamegen-dropdown__option--disabled': option.disabled
                }
              )}
              role="option"
              aria-selected={option.value === value}
              aria-disabled={option.disabled}
              onClick={() => handleSelect(option)}
              onMouseEnter={() => setFocusedIndex(index)}
              data-value={option.value}
              data-state={option.state}
              title={option.tooltip}
            >
              {option.icon && (
                <Icon
                  name={option.icon}
                  size={size === 'sm' ? 'sm' : 'md'}
                  className="gamegen-dropdown__option-icon"
                  ariaLabel=""
                />
              )}
              {option.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default Dropdown;