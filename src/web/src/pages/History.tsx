import React, { useCallback, useEffect, useState, useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { ErrorBoundary } from 'react-error-boundary';
import { useAnalytics } from '@segment/analytics-next';

import DashboardLayout from '../layouts/DashboardLayout';
import Loading from '../components/common/Loading';
import { GenerationService } from '../services/generation';
import { useAuth } from '../hooks/useAuth';
import { GenerationStatus } from '../types/generation';

// Interface for history item with comprehensive details
interface HistoryItem {
  id: string;
  prompt: string;
  status: GenerationStatus;
  timestamp: Date;
  metrics: {
    fid_score: number;
    fvd_score: number;
    generation_time_ms: number;
    actual_fps: number;
  };
  parameters: {
    resolution: {
      width: number;
      height: number;
    };
    frames: number;
    fps: number;
  };
  error?: string;
}

// Error fallback component
const ErrorFallback: React.FC<{ error: Error }> = ({ error }) => (
  <div 
    role="alert" 
    className="history-error"
    style={{
      padding: 'var(--spacing-lg)',
      color: 'rgb(var(--error-rgb))',
      textAlign: 'center'
    }}
  >
    <h2>Error Loading History</h2>
    <pre>{error.message}</pre>
  </div>
);

const History: React.FC = () => {
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [hasMore, setHasMore] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  const { analytics } = useAnalytics();
  const { checkPermission } = useAuth();

  // Virtual list configuration for performance
  const virtualizer = useVirtualizer({
    count: historyItems.length,
    getScrollElement: () => containerRef.current,
    estimateSize: () => 200,
    overscan: 5
  });

  // Load initial history data
  const loadHistory = useCallback(async (offset = 0) => {
    try {
      setIsLoading(true);
      const response = await GenerationService.getGenerationHistory(offset);
      setHistoryItems(prev => offset === 0 ? response : [...prev, ...response]);
      setHasMore(response.length === 20); // Assuming page size of 20
      analytics?.track('History Loaded', { offset, count: response.length });
    } catch (error) {
      console.error('Failed to load history:', error);
      analytics?.track('History Load Error', { error: error.message });
    } finally {
      setIsLoading(false);
    }
  }, [analytics]);

  // Setup infinite scroll
  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        if (entries[0].isIntersecting && hasMore && !isLoading) {
          loadHistory(historyItems.length);
        }
      },
      { threshold: 0.5 }
    );

    const sentinel = document.querySelector('.history-sentinel');
    if (sentinel) observer.observe(sentinel);

    return () => observer.disconnect();
  }, [hasMore, isLoading, historyItems.length, loadHistory]);

  // Handle retry functionality
  const handleRetry = useCallback(async (id: string) => {
    try {
      await GenerationService.retryGeneration(id);
      analytics?.track('Generation Retry', { generationId: id });
      loadHistory(0); // Reload history
    } catch (error) {
      console.error('Retry failed:', error);
      analytics?.track('Generation Retry Error', { 
        generationId: id, 
        error: error.message 
      });
    }
  }, [analytics, loadHistory]);

  // Setup WebSocket subscription for real-time updates
  useEffect(() => {
    const unsubscribe = GenerationService.subscribeToUpdates(
      (update) => {
        setHistoryItems(prev => prev.map(item => 
          item.id === update.id ? { ...item, ...update } : item
        ));
      }
    );

    return () => unsubscribe();
  }, []);

  // Render history item
  const renderHistoryItem = useCallback((item: HistoryItem) => (
    <div 
      className="history-item"
      style={{
        padding: 'var(--spacing-md)',
        borderRadius: 'var(--border-radius)',
        backgroundColor: 'var(--surface-rgb)',
        boxShadow: 'var(--shadow-sm)',
        transition: 'transform 0.2s ease',
        cursor: 'pointer'
      }}
      role="article"
      aria-labelledby={`generation-${item.id}-title`}
    >
      <h3 id={`generation-${item.id}-title`}>{item.prompt}</h3>
      
      <div className="history-item__details">
        <span>Status: {item.status}</span>
        <span>Resolution: {item.parameters.resolution.width}x{item.parameters.resolution.height}</span>
        <span>FPS: {item.parameters.fps}</span>
        <span>Duration: {(item.parameters.frames / item.parameters.fps).toFixed(1)}s</span>
      </div>

      <div className="history-item__metrics">
        <span>FID Score: {item.metrics.fid_score.toFixed(2)}</span>
        <span>FVD Score: {item.metrics.fvd_score.toFixed(2)}</span>
        <span>Generation Time: {(item.metrics.generation_time_ms / 1000).toFixed(2)}s</span>
        <span>Actual FPS: {item.metrics.actual_fps.toFixed(1)}</span>
      </div>

      {item.status === GenerationStatus.FAILED && (
        <button
          onClick={() => handleRetry(item.id)}
          className="history-item__retry"
          aria-label="Retry generation"
        >
          Retry
        </button>
      )}
    </div>
  ), [handleRetry]);

  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <DashboardLayout>
        <div 
          className="history-container"
          style={{
            padding: 'var(--spacing-lg)',
            maxWidth: '1200px',
            margin: '0 auto'
          }}
        >
          <h1>Generation History</h1>

          <div 
            ref={containerRef}
            className="history-list"
            style={{
              height: 'calc(100vh - 200px)',
              overflow: 'auto'
            }}
          >
            <div
              style={{
                height: `${virtualizer.getTotalSize()}px`,
                width: '100%',
                position: 'relative'
              }}
            >
              {virtualizer.getVirtualItems().map((virtualItem) => (
                <div
                  key={historyItems[virtualItem.index].id}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: `${virtualItem.size}px`,
                    transform: `translateY(${virtualItem.start}px)`
                  }}
                >
                  {renderHistoryItem(historyItems[virtualItem.index])}
                </div>
              ))}
            </div>

            {hasMore && (
              <div className="history-sentinel" aria-hidden="true" />
            )}
          </div>

          {isLoading && (
            <Loading 
              size="large" 
              label="Loading history..." 
            />
          )}
        </div>
      </DashboardLayout>
    </ErrorBoundary>
  );
};

export default History;