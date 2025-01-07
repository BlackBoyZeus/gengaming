import React, { StrictMode, Suspense } from 'react'; // ^18.2.0
import { createRoot } from 'react-dom/client'; // ^18.2.0
import { ErrorBoundary } from 'react-error-boundary'; // ^4.0.0
import { PerformanceMonitor } from '@performance-monitor/react'; // ^2.0.0
import { reportWebVitals } from 'web-vitals'; // ^3.0.0

import App from './App';
import { ThemeProvider } from './contexts/ThemeContext';
import { UI_CONSTANTS, SYSTEM_LIMITS } from './config/constants';

// Root element ID constant
const ROOT_ELEMENT_ID = 'root';

// Browser compatibility check based on technical specifications
const checkBrowserCompatibility = (): boolean => {
  const userAgent = navigator.userAgent.toLowerCase();
  
  // Chrome version >= 90
  const chromeMatch = /chrome\/(\d+)/.exec(userAgent);
  if (chromeMatch && parseInt(chromeMatch[1]) < 90) {
    return false;
  }

  // Firefox version >= 88
  const firefoxMatch = /firefox\/(\d+)/.exec(userAgent);
  if (firefoxMatch && parseInt(firefoxMatch[1]) < 88) {
    return false;
  }

  // Safari version >= 14
  const safariMatch = /version\/(\d+).*safari/.exec(userAgent);
  if (safariMatch && parseInt(safariMatch[1]) < 14) {
    return false;
  }

  return true;
};

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

// Initialize application with necessary checks and configurations
const initializeApp = (): void => {
  // Check browser compatibility
  if (!checkBrowserCompatibility()) {
    document.body.innerHTML = `
      <div style="text-align: center; padding: 2rem;">
        <h1>Browser Not Supported</h1>
        <p>Please use Chrome 90+, Firefox 88+, or Safari 14+</p>
      </div>
    `;
    return;
  }

  // Get root element
  const rootElement = document.getElementById(ROOT_ELEMENT_ID);
  if (!rootElement) {
    throw new Error(`Root element with id "${ROOT_ELEMENT_ID}" not found`);
  }

  // Create root with React 18 concurrent features
  const root = createRoot(rootElement);

  // Performance monitoring configuration
  const performanceConfig = {
    metrics: {
      fps: SYSTEM_LIMITS.MIN_FRAME_RATE,
      latency: SYSTEM_LIMITS.MAX_CONTROL_LATENCY,
      memory: parseInt(SYSTEM_LIMITS.MAX_MEMORY_USAGE)
    },
    onAlert: (metric: string, value: number) => {
      console.warn(`Performance alert: ${metric} = ${value}`);
    }
  };

  // Render application with all providers and monitoring
  root.render(
    <StrictMode>
      <ErrorBoundary FallbackComponent={ErrorFallback}>
        <PerformanceMonitor {...performanceConfig}>
          <ThemeProvider>
            <Suspense fallback={
              <div style={{ 
                height: '100vh', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center' 
              }}>
                Loading...
              </div>
            }>
              <App />
            </Suspense>
          </ThemeProvider>
        </PerformanceMonitor>
      </ErrorBoundary>
    </StrictMode>
  );

  // Report web vitals
  reportWebVitals(console.log);
};

// Initialize application
initializeApp();

// Enable hot module replacement in development
if (process.env.NODE_ENV === 'development' && module.hot) {
  module.hot.accept();
}