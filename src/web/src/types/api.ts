/**
 * Type definitions for GameGen-X API endpoints
 * @version 1.0.0
 */

import { GenerationParameters } from '../types/generation';

// API version constant
export const API_VERSION = 'v1';

/**
 * Enum defining types of video control inputs
 */
export enum ControlType {
    KEYBOARD = 'KEYBOARD',
    ENVIRONMENT = 'ENVIRONMENT',
    INSTRUCTION = 'INSTRUCTION'
}

/**
 * Enum defining video generation process status
 */
export enum GenerationStatus {
    PENDING = 'PENDING',
    PROCESSING = 'PROCESSING',
    COMPLETED = 'COMPLETED',
    FAILED = 'FAILED'
}

/**
 * Enum defining system health status indicators
 */
export enum SystemHealthStatus {
    HEALTHY = 'HEALTHY',
    DEGRADED = 'DEGRADED',
    UNHEALTHY = 'UNHEALTHY'
}

/**
 * Interface for video control request payload
 */
export interface ControlRequest {
    type: ControlType;
    data: Record<string, unknown>;
    generation_id: string;
    video_id: string;
    timestamp: string;
}

/**
 * Interface for video control response data
 */
export interface ControlResponse {
    id: string;
    type: ControlType;
    data: Record<string, unknown>;
    status: string;
    created_at: string;
    updated_at: string;
}

/**
 * Interface for video generation request payload
 */
export interface GenerationRequest {
    prompt: string;
    parameters: GenerationParameters;
    timestamp: string;
}

/**
 * Interface for video generation response data
 */
export interface GenerationResponse {
    id: string;
    prompt: string;
    parameters: GenerationParameters;
    status: GenerationStatus;
    error_details: Record<string, unknown>;
    metrics: Record<string, number>;
    created_at: string;
    updated_at: string;
}

/**
 * Interface for system resource utilization metrics
 */
export interface ResourceMetrics {
    cpu_usage_percent: number;
    memory_usage_percent: number;
    gpu_usage_percent: number;
    gpu_memory_percent: number;
    disk_usage_percent: number;
    network_bandwidth_mbps: number;
}

/**
 * Interface for system performance metrics
 */
export interface PerformanceMetrics {
    generation_latency_ms: number;
    frame_rate_fps: number;
    control_response_ms: number;
    concurrent_users: number;
    queue_depth: number;
    error_rate: number;
}

/**
 * Interface for comprehensive system status response
 */
export interface StatusResponse {
    system_status: SystemHealthStatus;
    resource_metrics: ResourceMetrics;
    performance_metrics: PerformanceMetrics;
    warnings: string[];
    errors: string[];
    timestamp: string;
}