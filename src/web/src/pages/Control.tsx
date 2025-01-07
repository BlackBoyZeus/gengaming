import React, { useCallback, useEffect, useState } from 'react';
import styled from '@emotion/styled';
import { ErrorBoundary } from 'react-error-boundary';
import { analytics } from '@segment/analytics-next';
import { usePerformanceMonitor } from 'performance-monitor';

import DashboardLayout from '../layouts/DashboardLayout';
import ControlPanel from '../components/control/ControlPanel';
import VideoViewport from '../components/video/VideoViewport';
import { useVideo } from '../hooks/useVideo';
import { SYSTEM_LIMITS, VIDEO_SETTINGS } from '../config/constants';

// Styled components with hardware acceleration and performance optimizations
const PageContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xl);
  height: 100%;
  padding: var(--spacing-lg);
  position: relative;
  isolation: isolate;
  transform: translateZ(0);
  will-change: transform;
  contain: layout style paint;
  
  @media (max-width: 768px) {
    padding: var(--spacing-md);
    gap: var(--spacing-lg);
  }
`;

const ViewportSection = styled.section`
  flex: 1;
  min-height: var(--video-viewport-min-height);
  background-color: rgb(var(--surface-rgb));
  border-radius: var(--border-radius-lg);
  overflow: hidden;
  position: relative;
  will-change: transform;
  contain: layout style paint;
`;

const ControlSection = styled.section`
  flex: 0 0 var(--control-panel-height);
  background-color: rgb(var(--surface-rgb));
  border-radius: var(--border-radius-lg);
  overflow: hidden;
  position: relative;
  contain: layout style paint;
`;

const Control: React.FC = () => {
  const { videoState, metrics } = useVideo();
  const [controlLatency, setControlLatency] = useState<number>(0);
  const [isPerformanceDegraded, setIsPerformanceDegraded] = useState(false);

  // Initialize performance monitoring
  const { startMeasurement, endMeasurement } = usePerformanceMonitor({
    metricName: 'control-response-time',
    threshold: SYSTEM_LIMITS.MAX_CONTROL_LATENCY
  });

  // Handle control changes with performance tracking
  const handleControlChange = useCallback(async (controlType: string, value: any) => {
    try {
      startMeasurement();
      const startTime = performance.now();

      // Track control interaction
      analytics.track('Control_Interaction', {
        controlType,
        value,
        videoState: videoState.status,
        metrics: {
          fps: metrics.frame_rate,
          latency: controlLatency
        }
      });

      // Process control change
      await new Promise(resolve => setTimeout(resolve, 0)); // Simulate control processing

      const latency = performance.now() - startTime;
      setControlLatency(latency);
      
      // Check for performance degradation
      if (latency > SYSTEM_LIMITS.MAX_CONTROL_LATENCY) {
        setIsPerformanceDegraded(true);
        console.warn(`Control latency exceeded threshold: ${latency.toFixed(2)}ms`);
      }

      endMeasurement();
    } catch (error) {
      console.error('Control change error:', error);
      analytics.track('Control_Error', {
        error: error.message,
        controlType,
        value
      });
    }
  }, [videoState.status, metrics, controlLatency, startMeasurement, endMeasurement]);

  // Handle errors with analytics tracking
  const handleError = useCallback((error: Error) => {
    console.error('Control page error:', error);
    analytics.track('Control_Page_Error', {
      error: error.message,
      videoState: videoState.status
    });
  }, [videoState.status]);

  // Monitor performance metrics
  useEffect(() => {
    const performanceObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach(entry => {
        if (entry.duration > SYSTEM_LIMITS.MAX_CONTROL_LATENCY) {
          setIsPerformanceDegraded(true);
        }
      });
    });

    performanceObserver.observe({ entryTypes: ['measure'] });
    return () => performanceObserver.disconnect();
  }, []);

  return (
    <ErrorBoundary
      FallbackComponent={({ error }) => (
        <div role="alert" aria-live="assertive">
          Error: {error.message}
        </div>
      )}
      onError={handleError}
    >
      <DashboardLayout>
        <PageContainer
          role="main"
          aria-label="Game control interface"
          data-performance-degraded={isPerformanceDegraded}
        >
          <ViewportSection>
            <VideoViewport
              width={VIDEO_SETTINGS.DEFAULT_RESOLUTION.width}
              height={VIDEO_SETTINGS.DEFAULT_RESOLUTION.height}
              onError={handleError}
            />
          </ViewportSection>

          <ControlSection>
            <ControlPanel
              enabled={videoState.isPlaying}
              onControlChange={handleControlChange}
              onError={handleError}
              onLatencyExceeded={(latency) => {
                setIsPerformanceDegraded(true);
                console.warn(`Control latency exceeded: ${latency}ms`);
              }}
            />
          </ControlSection>
        </PageContainer>
      </DashboardLayout>
    </ErrorBoundary>
  );
};

export default Control;