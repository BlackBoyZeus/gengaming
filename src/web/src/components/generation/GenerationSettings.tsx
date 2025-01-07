import React, { useState, useCallback, useEffect } from 'react';
import classnames from 'classnames'; // @version ^2.3.2
import { Dropdown } from '../common/Dropdown';
import { 
  Resolution, 
  Perspective,
  DEFAULT_RESOLUTION,
  DEFAULT_FRAMES,
  TARGET_FPS,
  MAX_FID_SCORE,
  MAX_FVD_SCORE
} from '../../types/generation';
import { useGeneration } from '../../hooks/useGeneration';
import { VIDEO_SETTINGS } from '../../config/constants';

// Resolution options based on technical specifications
const RESOLUTION_OPTIONS = VIDEO_SETTINGS.SUPPORTED_RESOLUTIONS.map(res => ({
  value: `${res.width}x${res.height}`,
  label: `${res.width}x${res.height} (${res.maxFrames} frames max)`,
  resolution: { width: res.width, height: res.height }
}));

// Frame count options based on technical specifications
const FRAME_OPTIONS = [
  { value: 102, label: '102 frames (~4.25s @ 24fps)' }
];

// Perspective options based on technical specifications
const PERSPECTIVE_OPTIONS = [
  { value: Perspective.FIRST_PERSON, label: 'First Person' },
  { value: Perspective.THIRD_PERSON, label: 'Third Person' }
];

interface GenerationSettingsProps {
  onSettingsChange: (parameters: GenerationParameters, isValid: boolean) => void;
  disabled?: boolean;
  className?: string;
  onValidationError?: (error: ValidationError) => void;
}

interface ValidationError {
  type: 'resolution' | 'frames' | 'quality';
  message: string;
}

interface GenerationParameters {
  resolution: Resolution;
  frames: number;
  perspective: Perspective;
  fps: number;
}

export const GenerationSettings: React.FC<GenerationSettingsProps> = ({
  onSettingsChange,
  disabled = false,
  className,
  onValidationError
}) => {
  // Initialize state with default values
  const [resolution, setResolution] = useState<Resolution>(DEFAULT_RESOLUTION);
  const [frames, setFrames] = useState<number>(DEFAULT_FRAMES);
  const [perspective, setPerspective] = useState<Perspective>(Perspective.THIRD_PERSON);
  const [isValid, setIsValid] = useState(true);

  // Access generation context for quality validation
  const { generationState, validateQuality } = useGeneration();

  // Validate settings and update parent component
  const validateAndUpdateSettings = useCallback(async () => {
    const parameters: GenerationParameters = {
      resolution,
      frames,
      perspective,
      fps: TARGET_FPS
    };

    let validationError: ValidationError | null = null;

    // Validate resolution
    const selectedResOption = RESOLUTION_OPTIONS.find(
      opt => opt.resolution.width === resolution.width && 
             opt.resolution.height === resolution.height
    );
    
    if (!selectedResOption) {
      validationError = {
        type: 'resolution',
        message: 'Invalid resolution selected'
      };
    }

    // Validate frame count
    const maxFrames = selectedResOption?.resolution ? 
      VIDEO_SETTINGS.SUPPORTED_RESOLUTIONS.find(
        res => res.width === selectedResOption.resolution.width
      )?.maxFrames : 0;

    if (maxFrames && frames > maxFrames) {
      validationError = {
        type: 'frames',
        message: `Maximum ${maxFrames} frames allowed for selected resolution`
      };
    }

    // Validate quality metrics if generation is active
    if (generationState) {
      const { metrics } = generationState;
      if (metrics.fid_score > MAX_FID_SCORE || metrics.fvd_score > MAX_FVD_SCORE) {
        validationError = {
          type: 'quality',
          message: 'Generation quality below required thresholds'
        };
      }
    }

    // Update validation state
    const newIsValid = !validationError;
    setIsValid(newIsValid);

    // Notify parent of validation error
    if (validationError && onValidationError) {
      onValidationError(validationError);
    }

    // Update parent with settings
    onSettingsChange(parameters, newIsValid);
  }, [resolution, frames, perspective, generationState, onSettingsChange, onValidationError]);

  // Handle resolution selection
  const handleResolutionChange = useCallback((value: string) => {
    const [width, height] = value.split('x').map(Number);
    setResolution({ width, height });
  }, []);

  // Handle frame count selection
  const handleFramesChange = useCallback((value: number) => {
    setFrames(value);
  }, []);

  // Handle perspective selection
  const handlePerspectiveChange = useCallback((value: string) => {
    setPerspective(value as Perspective);
  }, []);

  // Validate settings on any change
  useEffect(() => {
    validateAndUpdateSettings();
  }, [resolution, frames, perspective, validateAndUpdateSettings]);

  return (
    <div 
      className={classnames(
        'gamegen-settings',
        { 'gamegen-settings--disabled': disabled },
        { 'gamegen-settings--invalid': !isValid },
        className
      )}
      role="group"
      aria-label="Generation Settings"
    >
      <div className="gamegen-settings__field">
        <label 
          htmlFor="resolution-select"
          className="gamegen-settings__label"
        >
          Resolution
        </label>
        <Dropdown
          id="resolution-select"
          options={RESOLUTION_OPTIONS}
          value={`${resolution.width}x${resolution.height}`}
          onChange={handleResolutionChange}
          disabled={disabled}
          aria-label="Select video resolution"
          gameState={isValid ? 'idle' : 'error'}
        />
      </div>

      <div className="gamegen-settings__field">
        <label 
          htmlFor="frames-select"
          className="gamegen-settings__label"
        >
          Frame Count
        </label>
        <Dropdown
          id="frames-select"
          options={FRAME_OPTIONS}
          value={frames}
          onChange={handleFramesChange}
          disabled={disabled}
          aria-label="Select frame count"
          gameState={isValid ? 'idle' : 'error'}
        />
      </div>

      <div className="gamegen-settings__field">
        <label 
          htmlFor="perspective-select"
          className="gamegen-settings__label"
        >
          Camera Perspective
        </label>
        <Dropdown
          id="perspective-select"
          options={PERSPECTIVE_OPTIONS}
          value={perspective}
          onChange={handlePerspectiveChange}
          disabled={disabled}
          aria-label="Select camera perspective"
          gameState={isValid ? 'idle' : 'error'}
        />
      </div>
    </div>
  );
};

export default GenerationSettings;