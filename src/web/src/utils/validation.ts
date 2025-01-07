import { z } from 'zod';
import { GenerationParameters, Resolution, Perspective } from '../types/generation';
import { ControlRequest, ControlType } from '../types/api';

// Constants for validation bounds
const MIN_PROMPT_LENGTH = 10;
const MAX_PROMPT_LENGTH = 300;
const MIN_RESOLUTION = { width: 320, height: 256 };
const MAX_RESOLUTION = { width: 1280, height: 720 };
const SUPPORTED_ASPECT_RATIOS = ['4:3', '16:9'] as const;
const PROHIBITED_PATTERNS = [
  '<script>',
  'javascript:',
  'data:',
  'vbscript:',
  'on\\w+\\s*=',
  'style\\s*=',
];

// Zod schemas for validation
const promptSchema = z.string()
  .min(MIN_PROMPT_LENGTH, `Prompt must be at least ${MIN_PROMPT_LENGTH} characters`)
  .max(MAX_PROMPT_LENGTH, `Prompt must not exceed ${MAX_PROMPT_LENGTH} characters`)
  .refine(text => !PROHIBITED_PATTERNS.some(pattern => 
    new RegExp(pattern, 'i').test(text)
  ), 'Prompt contains prohibited patterns');

const resolutionSchema = z.object({
  width: z.number()
    .int('Width must be an integer')
    .min(MIN_RESOLUTION.width)
    .max(MAX_RESOLUTION.width),
  height: z.number()
    .int('Height must be an integer')
    .min(MIN_RESOLUTION.height)
    .max(MAX_RESOLUTION.height)
});

const generationParametersSchema = z.object({
  resolution: resolutionSchema,
  frames: z.number()
    .int('Frame count must be an integer')
    .min(24, 'Minimum 24 frames required')
    .max(102, 'Maximum 102 frames allowed'),
  perspective: z.nativeEnum(Perspective),
  fps: z.number()
    .int('FPS must be an integer')
    .min(24, 'Minimum 24 FPS required')
    .max(60, 'Maximum 60 FPS allowed')
});

const controlRequestSchema = z.object({
  type: z.nativeEnum(ControlType),
  data: z.record(z.unknown()),
  generation_id: z.string().uuid(),
  video_id: z.string().uuid(),
  timestamp: z.string().datetime()
});

/**
 * Validates text prompt for video generation
 * @param prompt - Text prompt to validate
 * @returns Result indicating validation success or error message
 */
export function validatePrompt(prompt: string): Result<boolean, string> {
  try {
    const sanitizedPrompt = prompt.trim();
    const result = promptSchema.safeParse(sanitizedPrompt);
    
    if (!result.success) {
      return { success: false, error: result.error.message };
    }
    
    return { success: true, data: true };
  } catch (error) {
    return { success: false, error: 'Invalid prompt format' };
  }
}

/**
 * Validates video resolution dimensions
 * @param width - Video width in pixels
 * @param height - Video height in pixels
 * @returns Result indicating validation success or error message
 */
export function validateResolution(width: number, height: number): Result<boolean, string> {
  try {
    const result = resolutionSchema.safeParse({ width, height });
    
    if (!result.success) {
      return { success: false, error: result.error.message };
    }

    // Check aspect ratio
    const ratio = width / height;
    const isValidRatio = SUPPORTED_ASPECT_RATIOS.some(supported => {
      const [w, h] = supported.split(':').map(Number);
      return Math.abs(ratio - (w / h)) < 0.01;
    });

    if (!isValidRatio) {
      return { success: false, error: 'Unsupported aspect ratio' };
    }

    return { success: true, data: true };
  } catch (error) {
    return { success: false, error: 'Invalid resolution format' };
  }
}

/**
 * Validates video generation parameters
 * @param params - Generation parameters to validate
 * @returns Result indicating validation success or error message
 */
export function validateGenerationParameters(params: GenerationParameters): Result<boolean, string> {
  try {
    const result = generationParametersSchema.safeParse(params);
    
    if (!result.success) {
      return { success: false, error: result.error.message };
    }

    // Validate resolution
    const resolutionResult = validateResolution(
      params.resolution.width,
      params.resolution.height
    );
    if (!resolutionResult.success) {
      return resolutionResult;
    }

    // Performance requirements check
    const totalPixels = params.resolution.width * params.resolution.height;
    const framesPerSecond = params.frames / (params.frames / params.fps);
    const pixelsPerSecond = totalPixels * framesPerSecond;

    if (pixelsPerSecond > 2073600) { // 1280x720x24fps limit
      return { 
        success: false, 
        error: 'Parameters exceed performance requirements'
      };
    }

    return { success: true, data: true };
  } catch (error) {
    return { success: false, error: 'Invalid generation parameters' };
  }
}

/**
 * Validates control request data
 * @param request - Control request to validate
 * @returns Result indicating validation success or error message
 */
export function validateControlRequest(request: ControlRequest): Result<boolean, string> {
  try {
    const result = controlRequestSchema.safeParse(request);
    
    if (!result.success) {
      return { success: false, error: result.error.message };
    }

    // Type-specific validation
    switch (request.type) {
      case ControlType.KEYBOARD:
        if (!request.data.key || typeof request.data.key !== 'string') {
          return { success: false, error: 'Invalid keyboard control data' };
        }
        break;
      case ControlType.ENVIRONMENT:
        if (!request.data.setting || typeof request.data.setting !== 'string') {
          return { success: false, error: 'Invalid environment control data' };
        }
        break;
      case ControlType.INSTRUCTION:
        if (!request.data.instruction || typeof request.data.instruction !== 'string') {
          return { success: false, error: 'Invalid instruction control data' };
        }
        break;
    }

    return { success: true, data: true };
  } catch (error) {
    return { success: false, error: 'Invalid control request' };
  }
}

// Type definition for validation results
interface Result<T, E = string> {
  success: boolean;
  data?: T;
  error?: E;
}