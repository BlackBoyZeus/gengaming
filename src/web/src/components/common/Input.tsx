import React, { useCallback, useMemo } from 'react';
import classnames from 'classnames';
import { validatePrompt } from '../../utils/validation';

// react: ^18.0.0
// classnames: ^2.3.2

interface InputProps {
  type?: 'text' | 'number' | 'password' | 'email';
  id: string;
  value: string | number;
  placeholder?: string;
  label: string;
  error?: string;
  disabled?: boolean;
  required?: boolean;
  autoFocus?: boolean;
  className?: string;
  onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onBlur?: (event: React.FocusEvent<HTMLInputElement>) => void;
  onFocus?: (event: React.FocusEvent<HTMLInputElement>) => void;
}

const Input: React.FC<InputProps> = React.memo(({
  type = 'text',
  id,
  value,
  placeholder,
  label,
  error,
  disabled = false,
  required = false,
  autoFocus = false,
  className,
  onChange,
  onBlur,
  onFocus
}) => {
  // Memoize validation state
  const validationState = useMemo(() => {
    if (type === 'text' && typeof value === 'string') {
      return validatePrompt(value);
    }
    return { success: true };
  }, [value, type]);

  // Debounced change handler with validation
  const handleChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    onChange(event);
  }, [onChange]);

  // Memoize class composition
  const inputClasses = useMemo(() => classnames(
    'input-base',
    {
      'input-error': error || !validationState.success,
      'input-disabled': disabled,
    },
    className
  ), [error, validationState.success, disabled, className]);

  const labelClasses = useMemo(() => classnames(
    'input-label',
    { 'label-error': error || !validationState.success }
  ), [error, validationState.success]);

  return (
    <div className="input-container">
      <label 
        htmlFor={id}
        className={labelClasses}
        data-required={required}
      >
        {label}
        {required && <span className="required-indicator" aria-hidden="true">*</span>}
      </label>
      
      <input
        id={id}
        type={type}
        value={value}
        onChange={handleChange}
        onBlur={onBlur}
        onFocus={onFocus}
        placeholder={placeholder}
        disabled={disabled}
        required={required}
        autoFocus={autoFocus}
        className={inputClasses}
        aria-invalid={!!error || !validationState.success}
        aria-required={required}
        aria-describedby={error ? `${id}-error` : undefined}
      />

      {(error || !validationState.success) && (
        <div 
          id={`${id}-error`}
          className="input-error-message"
          role="alert"
        >
          {error || validationState.error}
        </div>
      )}

      <style jsx>{`
        .input-container {
          display: flex;
          flex-direction: column;
          gap: var(--spacing-xs);
          font-family: var(--font-family-system);
          width: 100%;
        }

        .input-label {
          font-size: var(--font-size-sm);
          font-weight: 500;
          color: var(--text-color);
        }

        .label-error {
          color: var(--error-color);
        }

        .required-indicator {
          margin-left: var(--spacing-xs);
          color: var(--error-color);
        }

        .input-base {
          padding: var(--spacing-sm) var(--spacing-md);
          font-size: var(--font-size-base);
          border-radius: var(--border-radius);
          border: 2px solid var(--border-color);
          background-color: var(--background-color);
          color: var(--text-color);
          transition: var(--transition-speed);
          width: 100%;
          outline: none;
        }

        .input-base:focus {
          border-color: var(--primary-color);
          box-shadow: 0 0 0 2px var(--primary-color-alpha);
        }

        .input-error {
          border-color: var(--error-color);
          background-color: var(--error-bg-color);
        }

        .input-disabled {
          opacity: 0.6;
          cursor: not-allowed;
          background-color: var(--disabled-bg-color);
        }

        .input-error-message {
          font-size: var(--font-size-sm);
          color: var(--error-color);
          margin-top: var(--spacing-xs);
        }
      `}</style>
    </div>
  );
});

Input.displayName = 'Input';

export type { InputProps };
export default Input;