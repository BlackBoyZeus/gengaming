/**
 * Type definitions for GameGen-X video generation system
 * @version 1.0.0
 */

// Global constants for system configuration
export const DEFAULT_RESOLUTION = {
    width: 1280,  // 720p width
    height: 720   // 720p height
} as const;

export const DEFAULT_FRAMES = 102;  // Default number of frames per generation
export const TARGET_FPS = 24;       // Target frames per second
export const MAX_FID_SCORE = 300;   // Maximum acceptable FID score for quality
export const MAX_FVD_SCORE = 1000;  // Maximum acceptable FVD score for quality

/**
 * Interface defining video resolution configuration
 */
export interface Resolution {
    width: number;   // Video width in pixels
    height: number;  // Video height in pixels
}

/**
 * Enum defining available camera perspective options
 */
export enum Perspective {
    FIRST_PERSON = "FIRST_PERSON",
    THIRD_PERSON = "THIRD_PERSON"
}

/**
 * Enum defining possible states of the generation process
 */
export enum GenerationStatus {
    PENDING = "PENDING",         // Generation request created but not started
    GENERATING = "GENERATING",   // Generation actively in progress
    COMPLETED = "COMPLETED",     // Generation successfully completed
    FAILED = "FAILED"           // Generation failed with error
}

/**
 * Interface defining parameters for video generation
 */
export interface GenerationParameters {
    resolution: Resolution;     // Video resolution configuration
    frames: number;            // Number of frames to generate
    perspective: Perspective;   // Camera perspective setting
    fps: number;              // Target frames per second
}

/**
 * Interface defining quality and performance metrics for generation
 */
export interface GenerationMetrics {
    fid_score: number;          // Fréchet Inception Distance score
    fvd_score: number;          // Fréchet Video Distance score
    generation_time_ms: number; // Total generation time in milliseconds
    actual_fps: number;         // Actual achieved frames per second
}

/**
 * Interface defining the complete state of a generation process
 */
export interface GenerationState {
    id: string;                      // Unique identifier for the generation
    prompt: string;                  // Text prompt for generation
    parameters: GenerationParameters; // Generation configuration parameters
    status: GenerationStatus;        // Current status of generation
    progress: number;                // Progress percentage (0-100)
    metrics: GenerationMetrics;      // Quality and performance metrics
    error: string | null;            // Error message if generation failed
}