import React, { useCallback, useEffect, useState, useRef } from 'react';
import classnames from 'classnames'; // @version ^2.3.2

import Header from '../components/dashboard/Header';
import Sidebar from '../components/dashboard/Sidebar';
import { useAuth } from '../hooks/useAuth';
import { useWebSocket } from '../hooks/useWebSocket';
import { UI_CONSTANTS } from '../config/constants';
import { WS_MESSAGE_TYPES } from '../config/websocket';

// Props interface for the DashboardLayout component
export interface DashboardLayoutProps {
  children: React.ReactNode;
  theme?: 'light' | 'dark' | 'system';
  wsEndpoint?: string;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  children,
  theme = UI_CONSTANTS.THEME.SYSTEM,
  wsEndpoint = '/ws/stream'
}) => {
  // State management
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [highContrast, setHighContrast] = useState(false);
  const layoutRef = useRef<HTMLDivElement>(null);

  // Hooks
  const { authState } = useAuth();
  const { isConnected, latency, fps } = useWebSocket(wsEndpoint, {}, {
    onMessage: (message) => {
      if (message.type === WS_MESSAGE_TYPES.STATUS) {
        // Handle real-time status updates
        console.debug('WebSocket Status:', message.data);
      }
    },
    onError: (error) => {
      console.error('WebSocket Error:', error);
    }
  });

  // Handle sidebar toggle with performance optimization
  const handleSidebarToggle = useCallback(() => {
    if (layoutRef.current) {
      layoutRef.current.style.willChange = 'transform';
      requestAnimationFrame(() => {
        setIsSidebarCollapsed(prev => !prev);
        requestAnimationFrame(() => {
          if (layoutRef.current) {
            layoutRef.current.style.willChange = 'auto';
          }
        });
      });
    }
  }, []);

  // Setup keyboard shortcuts for gaming controls
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === '\\') {
        handleSidebarToggle();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [handleSidebarToggle]);

  // Handle system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleThemeChange = (e: MediaQueryListEvent) => {
      if (theme === 'system') {
        document.documentElement.dataset.theme = e.matches ? 'dark' : 'light';
      }
    };

    mediaQuery.addEventListener('change', handleThemeChange);
    return () => mediaQuery.removeEventListener('change', handleThemeChange);
  }, [theme]);

  // Base styles with hardware acceleration
  const styles = {
    container: {
      display: 'flex',
      flexDirection: 'column' as const,
      minHeight: '100vh',
      backgroundColor: 'var(--background-color)',
      transition: 'background-color var(--transition-speed-normal)',
      willChange: 'transform',
      contain: 'layout style paint',
      colorScheme: theme === 'system' ? 'light dark' : theme
    },
    main: {
      display: 'flex',
      flex: '1',
      marginTop: '64px',
      transform: 'translateZ(0)',
      backfaceVisibility: 'hidden' as const
    },
    content: {
      flex: '1',
      padding: 'var(--spacing-lg)',
      marginLeft: isSidebarCollapsed ? '64px' : '280px',
      transition: 'margin-left var(--transition-speed-normal) var(--transition-timing)',
      willChange: 'margin-left',
      contain: 'layout style paint'
    }
  };

  return (
    <div
      ref={layoutRef}
      className={classnames('gamegen-dashboard', {
        'gamegen-dashboard--high-contrast': highContrast
      })}
      style={styles.container}
      data-theme={theme}
      data-high-contrast={highContrast}
      role="application"
      aria-label="Game dashboard"
    >
      <Header
        theme={{
          mode: theme === 'system' ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : theme,
          highContrast
        }}
        performance={{
          reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
          lowLatency: latency < 100
        }}
      />

      <div style={styles.main}>
        <Sidebar
          isCollapsed={isSidebarCollapsed}
          onToggle={handleSidebarToggle}
          theme={theme === 'system' ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : theme}
          highContrast={highContrast}
        />

        <main
          style={styles.content}
          role="main"
          aria-label="Main content"
        >
          {/* Performance metrics display */}
          {process.env.NODE_ENV === 'development' && (
            <div
              role="status"
              aria-label="Performance metrics"
              style={{
                position: 'fixed',
                bottom: 16,
                right: 16,
                padding: 8,
                backgroundColor: 'var(--surface-color)',
                borderRadius: 4,
                fontSize: 12,
                opacity: 0.8
              }}
            >
              <div>WebSocket: {isConnected ? 'Connected' : 'Disconnected'}</div>
              <div>Latency: {latency}ms</div>
              <div>FPS: {fps}</div>
            </div>
          )}

          {children}
        </main>
      </div>
    </div>
  );
};

export default DashboardLayout;