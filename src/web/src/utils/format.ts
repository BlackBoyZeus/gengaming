import { format } from 'date-fns';
import { GenerationMetrics } from '../types/generation';
import { VideoMetrics } from '../types/video';

/**
 * Formats and validates resolution dimensions with 720p requirement check
 * @param width - Video width in pixels
 * @param height - Video height in pixels
 * @returns Formatted resolution string with validation status
 */
export function formatResolution(width: number, height: number): string {
  // Validate inputs are positive integers
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error('Resolution dimensions must be positive integers');
  }

  // Check if resolution meets 720p minimum (1280x720)
  const meets720p = width >= 1280 && height >= 720;
  const formattedDimensions = `${width}x${height}`;

  return meets720p ? formattedDimensions : `${formattedDimensions} (below 720p)`;
}

/**
 * Formats frame rate with target validation (24 FPS)
 * @param fps - Frames per second value
 * @returns Formatted FPS string with target indicator
 */
export function formatFrameRate(fps: number): string {
  if (fps <= 0) {
    throw new Error('Frame rate must be a positive number');
  }

  // Use different precision based on FPS value
  const formattedFps = fps < 24 ? fps.toFixed(2) : Math.round(fps).toString();
  const targetIndicator = fps >= 24 ? ' ✓' : ' ⚠';

  return `${formattedFps} FPS${targetIndicator}`;
}

/**
 * Formats video duration in seconds to MM:SS format
 * @param seconds - Duration in seconds
 * @returns Formatted duration string
 */
export function formatDuration(seconds: number): string {
  if (seconds < 0) {
    throw new Error('Duration must be non-negative');
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);

  // Handle overflow case
  if (minutes > 59) {
    return '59:59+';
  }

  // Pad with leading zeros
  const paddedMinutes = minutes.toString().padStart(2, '0');
  const paddedSeconds = remainingSeconds.toString().padStart(2, '0');

  return `${paddedMinutes}:${paddedSeconds}`;
}

/**
 * Formats generation quality metrics with threshold validation
 * @param metrics - Generation metrics object
 * @returns Formatted metrics with validation status
 */
export function formatGenerationMetrics(metrics: GenerationMetrics): {
  fid: string;
  fvd: string;
  time: string;
} {
  const {
    fid_score,
    fvd_score,
    generation_time_ms
  } = metrics;

  // Validate FID score (threshold: 300)
  const fidStatus = fid_score <= 300 ? '✓' : '⚠';
  const formattedFid = `${fid_score.toFixed(2)} ${fidStatus}`;

  // Validate FVD score (threshold: 1000)
  const fvdStatus = fvd_score <= 1000 ? '✓' : '⚠';
  const formattedFvd = `${fvd_score.toFixed(2)} ${fvdStatus}`;

  // Convert generation time to seconds with 2 decimal precision
  const timeSeconds = (generation_time_ms / 1000).toFixed(2);
  const formattedTime = `${timeSeconds}s`;

  return {
    fid: formattedFid,
    fvd: formattedFvd,
    time: formattedTime
  };
}

/**
 * Formats ISO timestamp to localized date-time string
 * @param timestamp - ISO format timestamp string
 * @returns Localized date-time string
 */
export function formatTimestamp(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    
    // Validate date parsing
    if (isNaN(date.getTime())) {
      throw new Error('Invalid date');
    }

    // Format with date-fns using ISO format with timezone
    return format(date, 'yyyy-MM-dd HH:mm:ss XXX');
  } catch (error) {
    console.error('Error formatting timestamp:', error);
    return 'Invalid timestamp';
  }
}