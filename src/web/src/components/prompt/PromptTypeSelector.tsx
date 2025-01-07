import React, { useCallback, useMemo } from 'react'; // ^18.0.0
import styled from '@emotion/styled'; // ^11.0.0
import RadioGroup, { RadioGroupProps, RadioOption } from '../common/RadioGroup';

// Enums
export enum PromptType {
  CANNY_EDGE = 'canny_edge',
  MOTION_VECTORS = 'motion_vectors',
  POSE_SEQUENCE = 'pose_sequence'
}

// Interfaces
export interface PromptTypeSelectorProps {
  selectedType: PromptType;
  onChange: (type: PromptType) => void;
  disabled?: boolean;
  ariaLabel?: string;
}

// Styled Components
const SelectorContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: var(--spacing-unit);
  padding: var(--spacing-unit);
  background: var(--background-secondary);
  border-radius: var(--border-radius);
  position: relative;

  &:focus-within {
    outline: 2px solid var(--focus-ring);
  }

  @media (max-width: 768px) {
    padding: var(--spacing-unit-half);
  }
`;

const SelectorLabel = styled.label`
  font-size: var(--font-size-base);
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: var(--spacing-unit-half);
  user-select: none;

  @media (max-width: 768px) {
    font-size: var(--font-size-sm);
  }
`;

const ErrorMessage = styled.div`
  color: var(--error-color);
  font-size: var(--font-size-sm);
  margin-top: var(--spacing-unit-half);
  padding: var(--spacing-unit-half);
  background-color: var(--error-background);
  border-radius: var(--border-radius);
  display: flex;
  align-items: center;
  gap: var(--spacing-unit-half);
`;

// Component
const PromptTypeSelector: React.FC<PromptTypeSelectorProps> = React.memo(({
  selectedType,
  onChange,
  disabled = false,
  ariaLabel = 'Select prompt type'
}) => {
  // Convert enum to radio options
  const promptOptions = useMemo((): RadioOption[] => [
    {
      value: PromptType.CANNY_EDGE,
      label: 'Canny Edge Sequence'
    },
    {
      value: PromptType.MOTION_VECTORS,
      label: 'Motion Vectors'
    },
    {
      value: PromptType.POSE_SEQUENCE,
      label: 'Pose Sequence'
    }
  ], []);

  // Handle prompt type changes with validation
  const handlePromptTypeChange = useCallback((value: string) => {
    try {
      // Validate that the value is a valid PromptType
      if (Object.values(PromptType).includes(value as PromptType)) {
        onChange(value as PromptType);
      } else {
        console.error(`Invalid prompt type: ${value}`);
        // Update ARIA live region for screen readers
        const liveRegion = document.getElementById('prompt-type-error');
        if (liveRegion) {
          liveRegion.textContent = 'Invalid prompt type selected';
        }
      }
    } catch (error) {
      console.error('Error handling prompt type change:', error);
    }
  }, [onChange]);

  return (
    <SelectorContainer
      role="region"
      aria-labelledby="prompt-type-label"
    >
      <SelectorLabel id="prompt-type-label">
        {ariaLabel}
      </SelectorLabel>

      <RadioGroup
        name="prompt-type"
        options={promptOptions}
        value={selectedType}
        onChange={handlePromptTypeChange}
        disabled={disabled}
        className="prompt-type-radio-group"
      />

      {/* Hidden live region for screen reader announcements */}
      <div
        id="prompt-type-error"
        role="alert"
        aria-live="polite"
        className="visually-hidden"
      />

      {/* Error boundary fallback */}
      <React.Suspense fallback={null}>
        {({ error }: { error?: Error }) => (
          error && (
            <ErrorMessage role="alert">
              <span aria-hidden="true">⚠️</span>
              An error occurred while selecting the prompt type
            </ErrorMessage>
          )
        )}
      </React.Suspense>
    </SelectorContainer>
  );
});

PromptTypeSelector.displayName = 'PromptTypeSelector';

export default PromptTypeSelector;