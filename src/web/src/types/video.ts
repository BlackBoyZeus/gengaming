/**
 * Type definitions for video-related data structures in GameGen-X
 * @version 1.0.0
 */

import { GenerationStatus } from './api';

/**
 * Enum defining supported video formats
 */
export enum VideoFormat {
    MP4 = 'MP4',
    WEBM = 'WEBM'
}

/**
 * Interface defining comprehensive video quality and performance metrics
 * Tracks key performance indicators defined in technical specifications
 */
export interface VideoMetrics {
    frame_rate: number;           // Current frame rate (target: 24 FPS)
    resolution_width: number;     // Video width in pixels (target: 1280)
    resolution_height: number;    // Video height in pixels (target: 720)
    duration_seconds: number;     // Total video duration
    latency_ms: number;          // Processing latency (target: <100ms)
    quality_score: number;       // Overall quality score (0-100)
    buffer_size: number;         // Frame buffer size in bytes
    fid_score: number;          // Fréchet Inception Distance (target: <300)
    fvd_score: number;          // Fréchet Video Distance (target: <1000)
}

/**
 * Interface defining individual video frame data structure
 * Includes validation and metadata for frame-level quality control
 */
export interface VideoFrame {
    id: string;                           // Unique frame identifier
    sequence: number;                     // Frame sequence number
    data: Blob;                          // Frame binary data
    timestamp: number;                    // Frame timestamp in milliseconds
    metadata: Record<string, unknown>;    // Additional frame metadata
    validation_status: boolean;           // Frame validation status
}

/**
 * Interface defining comprehensive video data structure
 * Includes enhanced monitoring and error tracking capabilities
 */
export interface Video {
    id: string;                          // Unique video identifier
    generation_id: string;               // Associated generation request ID
    status: GenerationStatus;            // Current generation status
    format: VideoFormat;                 // Video format (MP4/WEBM)
    frame_count: number;                 // Total number of frames
    metrics: VideoMetrics;               // Performance and quality metrics
    created_at: string;                  // Creation timestamp
    updated_at: string;                  // Last update timestamp
    error_details: Record<string, unknown>; // Detailed error information
    processing_stats: Record<string, number>; // Processing statistics
    cache_status: Record<string, boolean>;   // Frame cache status
}