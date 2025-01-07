import React, { useCallback, useEffect } from 'react';
import { BrowserRouter, Routes, Route, useNavigationType } from 'react-router-dom'; // ^6.4.0
import { useHapticFeedback } from '@react-native-community/hooks'; // ^3.0.0
import { ErrorBoundary } from 'react-error-boundary'; // ^4.0.0
import { PerformanceMonitor } from '@performance-monitor/react'; // ^2.0.0

import MainLayout from './layouts/MainLayout';
import Home from './pages/Home';
import Control from './pages/Control';
import { ThemeProvider } from './contexts/ThemeContext';
import { UI_CONSTANTS, SYSTEM_LIMITS } from './config/constants';

// Error fallback component with gaming theme
const ErrorFallback: React.FC<{ error: Error }> = ({ error }) => (
  <div
    role="alert"
    style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: 'rgb(var(--background-rgb))',
      color: 'rgb(var(--error-rgb))',
      padding: 'var(--spacing-xl)',
      textAlign: 'center'
    }}
  >
    <div>
      <h1>Application Error</h1>
      <pre style={{ marginTop: 'var(--spacing-md)' }}>
        {process.env.NODE_ENV === 'development' ? error.message : 'Please refresh the page'}
      </pre>
      <button
        onClick={() => window.location.reload()}
        style={{
          marginTop: 'var(--spacing-lg)',
          padding: 'var(--spacing-md) var(--spacing-lg)',
          backgroundColor: 'rgb(var(--primary-rgb))',
          border: 'none',
          borderRadius: 'var(--border-radius-md)',
          color: 'rgb(var(--text-rgb))',
          cursor: 'pointer'
        }}
      >
        Retry
      </button>
    </div>
  </div>
);

// Performance monitoring wrapper
const withPerformanceMonitoring = (WrappedComponent: React.ComponentType) => {
  return function WithPerformanceMonitoring(props: any) {
    const navigationType = useNavigationType();

    useEffect(() => {
      performance.mark('route-change-start');
      return () => {
        performance.mark('route-change-end');
        performance.measure(
          'route-transition',
          'route-change-start',
          'route-change-end'
        );
      };
    }, [navigationType]);

    return <WrappedComponent {...props} />;
  };
};

// Main application component
const App: React.FC = () => {
  const { trigger } = useHapticFeedback();

  // Handle route transitions with haptic feedback
  const handleRouteTransition = useCallback(() => {
    trigger('impactLight');
  }, [trigger]);

  // Monitor performance metrics
  useEffect(() => {
    const observer = new PerformanceObserver((list) => {
      list.getEntries().forEach((entry) => {
        if (entry.duration > SYSTEM_LIMITS.MAX_CONTROL_LATENCY) {
          console.warn(`Performance degradation detected: ${entry.name}`);
        }
      });
    });

    observer.observe({ entryTypes: ['measure'] });
    return () => observer.disconnect();
  }, []);

  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <ThemeProvider>
        <BrowserRouter>
          <div
            style={{
              minHeight: '100vh',
              display: 'flex',
              flexDirection: 'column',
              transform: 'translateZ(0)',
              backfaceVisibility: 'hidden',
              perspective: 1000,
              willChange: 'transform'
            }}
          >
            <MainLayout>
              <Routes>
                <Route 
                  path="/" 
                  element={<Home />} 
                  handle={{ 
                    loadingStrategy: 'eager',
                    transitionThreshold: '100ms'
                  }} 
                />
                <Route 
                  path="/control/:generationId" 
                  element={<Control />}
                  handle={{ 
                    loadingStrategy: 'priority',
                    transitionThreshold: '50ms'
                  }}
                />
              </Routes>
            </MainLayout>
          </div>
        </BrowserRouter>
      </ThemeProvider>
    </ErrorBoundary>
  );
};

// Export performance-monitored app
export default withPerformanceMonitoring(App);