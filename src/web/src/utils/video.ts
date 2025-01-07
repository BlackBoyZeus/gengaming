/**
 * Video processing and performance optimization utilities for GameGen-X
 * @version 1.0.0
 */

import { VideoFormat, VideoMetrics, VideoFrame } from '../types/video';
import { TARGET_FPS, DEFAULT_RESOLUTION } from '../types/generation';

// Performance constants
const MAX_LATENCY_MS = 100;
const BUFFER_SIZE_LIMIT = 1024 * 1024 * 50; // 50MB buffer limit
const FRAME_DROP_THRESHOLD = 5; // Maximum consecutive frames to drop
const PERFORMANCE_SAMPLE_SIZE = 60; // Rolling window for performance metrics

/**
 * Calculates comprehensive frame metrics with high-precision timing
 * @param frame Current video frame
 * @param timestamp High-precision timestamp
 * @param bufferSize Current buffer size in bytes
 * @returns VideoMetrics object with performance data
 */
export function calculateFrameMetrics(
    frame: VideoFrame,
    timestamp: number,
    bufferSize: number
): VideoMetrics {
    const now = performance.now();
    const latency = now - frame.timestamp;
    
    return {
        frame_rate: 1000 / (timestamp - frame.timestamp),
        resolution_width: DEFAULT_RESOLUTION.width,
        resolution_height: DEFAULT_RESOLUTION.height,
        latency_ms: latency,
        buffer_size: bufferSize,
        quality_score: calculateQualityScore(latency, bufferSize),
        duration_seconds: timestamp / 1000,
        fid_score: 0, // Updated by ML pipeline
        fvd_score: 0  // Updated by ML pipeline
    };
}

/**
 * Processes video frames with memory optimization and error recovery
 * @param frameData Raw frame blob data
 * @param sequence Frame sequence number
 * @returns Promise resolving to processed VideoFrame
 */
export async function processVideoFrame(
    frameData: Blob,
    sequence: number
): Promise<VideoFrame> {
    try {
        // Validate frame size
        if (frameData.size > BUFFER_SIZE_LIMIT / PERFORMANCE_SAMPLE_SIZE) {
            throw new Error('Frame size exceeds buffer limits');
        }

        const frame: VideoFrame = {
            id: crypto.randomUUID(),
            sequence,
            data: frameData,
            timestamp: performance.now(),
            metadata: {
                size: frameData.size,
                optimized: false
            },
            validation_status: true
        };

        // Optimize frame if needed
        if (frameData.size > BUFFER_SIZE_LIMIT / (PERFORMANCE_SAMPLE_SIZE * 2)) {
            frame.data = await optimizeFrameData(frameData);
            frame.metadata.optimized = true;
        }

        return frame;
    } catch (error) {
        console.error('Frame processing error:', error);
        throw error;
    }
}

/**
 * Optimizes frame buffer for performance and memory usage
 * @param buffer Current frame buffer
 * @param targetFPS Desired frames per second
 * @param currentLatency Current processing latency
 * @returns Optimized frame buffer
 */
export function optimizeFrameBuffer(
    buffer: VideoFrame[],
    targetFPS: number = TARGET_FPS,
    currentLatency: number
): VideoFrame[] {
    // Calculate optimal buffer size based on performance targets
    const optimalSize = Math.ceil(targetFPS * (MAX_LATENCY_MS / currentLatency));
    
    // Sort frames by sequence
    buffer.sort((a, b) => a.sequence - b.sequence);
    
    // Remove excess frames if buffer is too large
    if (buffer.length > optimalSize) {
        const dropCount = buffer.length - optimalSize;
        if (dropCount <= FRAME_DROP_THRESHOLD) {
            buffer = buffer.slice(dropCount);
        } else {
            // Intelligent frame dropping to maintain visual quality
            buffer = dropFramesIntelligently(buffer, optimalSize);
        }
    }

    return buffer;
}

/**
 * Validates video format compatibility with browser capabilities
 * @param format Video format to validate
 * @param browserCapabilities Browser capability object
 * @returns Validation result with compatibility details
 */
export function validateVideoFormat(
    format: VideoFormat,
    browserCapabilities: { [key: string]: boolean }
): { 
    isValid: boolean;
    details: Record<string, unknown>;
} {
    const validation = {
        isValid: false,
        details: {
            format,
            supported: false,
            webgl: hasWebGLSupport(),
            resolution: checkResolutionSupport(),
            codec: checkCodecSupport(format)
        }
    };

    // Check format support
    switch (format) {
        case VideoFormat.MP4:
            validation.details.supported = browserCapabilities.mp4 !== false;
            break;
        case VideoFormat.WEBM:
            validation.details.supported = browserCapabilities.webm !== false;
            break;
    }

    validation.isValid = validation.details.supported &&
                        validation.details.webgl &&
                        validation.details.resolution &&
                        validation.details.codec;

    return validation;
}

// Private utility functions

/**
 * Calculates quality score based on performance metrics
 */
function calculateQualityScore(latency: number, bufferSize: number): number {
    const latencyScore = Math.max(0, 100 - (latency / MAX_LATENCY_MS) * 100);
    const bufferScore = Math.max(0, 100 - (bufferSize / BUFFER_SIZE_LIMIT) * 100);
    return Math.floor((latencyScore + bufferScore) / 2);
}

/**
 * Optimizes frame data size while maintaining quality
 */
async function optimizeFrameData(frameData: Blob): Promise<Blob> {
    // Implementation would include frame compression logic
    return frameData;
}

/**
 * Intelligently drops frames to maintain visual quality
 */
function dropFramesIntelligently(
    buffer: VideoFrame[],
    targetSize: number
): VideoFrame[] {
    const stride = Math.ceil(buffer.length / targetSize);
    return buffer.filter((_, index) => index % stride === 0);
}

/**
 * Checks WebGL support
 */
function hasWebGLSupport(): boolean {
    try {
        const canvas = document.createElement('canvas');
        return !!(
            window.WebGLRenderingContext &&
            (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
        );
    } catch {
        return false;
    }
}

/**
 * Validates resolution support
 */
function checkResolutionSupport(): boolean {
    return window.screen.width >= DEFAULT_RESOLUTION.width &&
           window.screen.height >= DEFAULT_RESOLUTION.height;
}

/**
 * Checks codec support for video format
 */
function checkCodecSupport(format: VideoFormat): boolean {
    const video = document.createElement('video');
    switch (format) {
        case VideoFormat.MP4:
            return video.canPlayType('video/mp4') !== '';
        case VideoFormat.WEBM:
            return video.canPlayType('video/webm') !== '';
        default:
            return false;
    }
}