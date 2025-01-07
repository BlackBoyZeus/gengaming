import { describe, test, expect } from '@jest/globals';
import { performance } from 'jest-performance';
import {
  validatePrompt,
  validateGenerationParameters,
  validateControlRequest,
  validateResolution
} from '../../src/utils/validation';
import { Perspective } from '../../src/types/generation';
import { ControlType } from '../../src/types/api';

// Test constants
const TEST_PROMPTS = {
  valid: 'Generate a medieval castle with high towers and a moat',
  tooShort: 'Short',
  tooLong: 'a'.repeat(301),
  withXSS: '<script>alert("xss")</script>',
  withJavaScript: 'javascript:alert(1)',
  withOnEvent: 'onclick=alert(1)',
  withStyle: 'style=background:url(evil.com)',
  withUnicode: '🏰 Generate a castle with 城堡',
  empty: '',
  nullValue: null,
  undefined: undefined
};

const TEST_RESOLUTIONS = {
  valid720p: { width: 1280, height: 720 },
  valid480p: { width: 848, height: 480 },
  valid320p: { width: 320, height: 256 },
  tooSmall: { width: 319, height: 255 },
  tooLarge: { width: 1281, height: 721 },
  invalidRatio: { width: 1000, height: 500 },
  nonInteger: { width: 320.5, height: 256.7 }
};

const TEST_PARAMETERS = {
  valid: {
    resolution: { width: 1280, height: 720 },
    frames: 102,
    perspective: Perspective.THIRD_PERSON,
    fps: 24
  },
  invalidFrames: {
    resolution: { width: 1280, height: 720 },
    frames: 103,
    perspective: Perspective.THIRD_PERSON,
    fps: 24
  },
  invalidFPS: {
    resolution: { width: 1280, height: 720 },
    frames: 102,
    perspective: Perspective.THIRD_PERSON,
    fps: 61
  }
};

const PERFORMANCE_THRESHOLDS = {
  validationTime: 50, // ms
  memoryIncrease: 1024 * 1024 // 1MB
};

describe('Prompt Validation Tests', () => {
  test('should validate correct prompts', () => {
    const result = validatePrompt(TEST_PROMPTS.valid);
    expect(result.success).toBe(true);
  });

  test('should reject prompts that are too short', () => {
    const result = validatePrompt(TEST_PROMPTS.tooShort);
    expect(result.success).toBe(false);
    expect(result.error).toContain('at least');
  });

  test('should reject prompts that are too long', () => {
    const result = validatePrompt(TEST_PROMPTS.tooLong);
    expect(result.success).toBe(false);
    expect(result.error).toContain('exceed');
  });

  test('should reject XSS attack patterns', () => {
    const result = validatePrompt(TEST_PROMPTS.withXSS);
    expect(result.success).toBe(false);
    expect(result.error).toContain('prohibited patterns');
  });

  test('should handle Unicode characters correctly', () => {
    const result = validatePrompt(TEST_PROMPTS.withUnicode);
    expect(result.success).toBe(true);
  });

  test('should validate prompts within performance thresholds', () => {
    const startTime = performance.now();
    validatePrompt(TEST_PROMPTS.valid);
    const endTime = performance.now();
    expect(endTime - startTime).toBeLessThan(PERFORMANCE_THRESHOLDS.validationTime);
  });
});

describe('Resolution Validation Tests', () => {
  test('should validate correct resolutions', () => {
    const result = validateResolution(TEST_RESOLUTIONS.valid720p.width, TEST_RESOLUTIONS.valid720p.height);
    expect(result.success).toBe(true);
  });

  test('should reject resolutions below minimum', () => {
    const result = validateResolution(TEST_RESOLUTIONS.tooSmall.width, TEST_RESOLUTIONS.tooSmall.height);
    expect(result.success).toBe(false);
  });

  test('should reject resolutions above maximum', () => {
    const result = validateResolution(TEST_RESOLUTIONS.tooLarge.width, TEST_RESOLUTIONS.tooLarge.height);
    expect(result.success).toBe(false);
  });

  test('should reject invalid aspect ratios', () => {
    const result = validateResolution(TEST_RESOLUTIONS.invalidRatio.width, TEST_RESOLUTIONS.invalidRatio.height);
    expect(result.success).toBe(false);
    expect(result.error).toContain('aspect ratio');
  });

  test('should reject non-integer dimensions', () => {
    const result = validateResolution(TEST_RESOLUTIONS.nonInteger.width, TEST_RESOLUTIONS.nonInteger.height);
    expect(result.success).toBe(false);
  });
});

describe('Generation Parameters Validation Tests', () => {
  test('should validate correct parameters', () => {
    const result = validateGenerationParameters(TEST_PARAMETERS.valid);
    expect(result.success).toBe(true);
  });

  test('should reject invalid frame counts', () => {
    const result = validateGenerationParameters(TEST_PARAMETERS.invalidFrames);
    expect(result.success).toBe(false);
    expect(result.error).toContain('frames');
  });

  test('should reject invalid FPS values', () => {
    const result = validateGenerationParameters(TEST_PARAMETERS.invalidFPS);
    expect(result.success).toBe(false);
    expect(result.error).toContain('FPS');
  });

  test('should validate performance requirements', () => {
    const params = {
      ...TEST_PARAMETERS.valid,
      resolution: { width: 1280, height: 720 },
      frames: 102,
      fps: 24
    };
    const result = validateGenerationParameters(params);
    expect(result.success).toBe(true);
  });
});

describe('Control Request Validation Tests', () => {
  test('should validate keyboard control requests', () => {
    const request = {
      type: ControlType.KEYBOARD,
      data: { key: 'W' },
      generation_id: '123e4567-e89b-12d3-a456-426614174000',
      video_id: '123e4567-e89b-12d3-a456-426614174001',
      timestamp: new Date().toISOString()
    };
    const result = validateControlRequest(request);
    expect(result.success).toBe(true);
  });

  test('should validate environment control requests', () => {
    const request = {
      type: ControlType.ENVIRONMENT,
      data: { setting: 'DAY' },
      generation_id: '123e4567-e89b-12d3-a456-426614174000',
      video_id: '123e4567-e89b-12d3-a456-426614174001',
      timestamp: new Date().toISOString()
    };
    const result = validateControlRequest(request);
    expect(result.success).toBe(true);
  });

  test('should reject invalid control types', () => {
    const request = {
      type: 'INVALID_TYPE',
      data: {},
      generation_id: '123e4567-e89b-12d3-a456-426614174000',
      video_id: '123e4567-e89b-12d3-a456-426614174001',
      timestamp: new Date().toISOString()
    };
    const result = validateControlRequest(request as any);
    expect(result.success).toBe(false);
  });

  test('should reject malformed control data', () => {
    const request = {
      type: ControlType.KEYBOARD,
      data: {},
      generation_id: '123e4567-e89b-12d3-a456-426614174000',
      video_id: '123e4567-e89b-12d3-a456-426614174001',
      timestamp: new Date().toISOString()
    };
    const result = validateControlRequest(request);
    expect(result.success).toBe(false);
  });
});