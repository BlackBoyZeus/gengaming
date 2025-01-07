import React, { useCallback, useEffect, useState, memo } from 'react';
import classnames from 'classnames'; // @version ^2.3.2
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { useAuth } from '../../hooks/useAuth';
import { UI_CONSTANTS, CONTROL_SETTINGS } from '../../config/constants';

// Interface for game mode state
interface GameMode {
  active: boolean;
  reducedMotion: boolean;
  highContrast: boolean;
}

// Props interface for the Dashboard component
interface DashboardProps {
  children: React.ReactNode;
  className?: string;
  gameMode?: GameMode;
  reducedMotion?: boolean;
}

// Base styles with hardware acceleration and containment
const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    minHeight: '100vh',
    backgroundColor: 'var(--background-rgb)',
    transform: 'translateZ(0)',
    backfaceVisibility: 'hidden',
    willChange: 'transform',
    contain: 'layout style paint',
  },
  main: {
    display: 'flex',
    flex: '1',
    marginTop: '64px',
    transform: 'translateZ(0)',
    willChange: 'transform',
    contain: 'layout',
  },
  content: {
    flex: '1',
    padding: 'var(--spacing-lg)',
    marginLeft: '280px',
    transition: 'margin-left var(--transition-speed-normal) var(--transition-timing)',
    transform: 'translateZ(0)',
    contain: 'layout paint',
  },
  contentCollapsed: {
    marginLeft: '64px',
  },
  gameMode: {
    cursor: 'none',
    userSelect: 'none' as const,
    WebkitUserSelect: 'none' as const,
  }
} as const;

// Dashboard component implementation
const Dashboard = memo(({
  children,
  className,
  gameMode = { active: false, reducedMotion: false, highContrast: false },
  reducedMotion = false,
}: DashboardProps) => {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>(UI_CONSTANTS.THEME.SYSTEM);
  const { authState } = useAuth();

  // Handle sidebar toggle with optimized animation
  const handleSidebarToggle = useCallback(() => {
    if (reducedMotion) {
      setIsSidebarCollapsed(prev => !prev);
      return;
    }

    requestAnimationFrame(() => {
      setIsSidebarCollapsed(prev => !prev);
    });
  }, [reducedMotion]);

  // Setup keyboard shortcuts for gaming mode
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === '\\') {
        handleSidebarToggle();
      }
    };

    if (gameMode.active) {
      window.addEventListener('keydown', handleKeyPress);
      return () => window.removeEventListener('keydown', handleKeyPress);
    }
  }, [gameMode.active, handleSidebarToggle]);

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

  // Compute container classes with performance optimizations
  const containerClasses = classnames(
    'gamegen-dashboard',
    {
      'gamegen-dashboard--game-mode': gameMode.active,
      'gamegen-dashboard--reduced-motion': reducedMotion || gameMode.reducedMotion,
      'gamegen-dashboard--high-contrast': gameMode.highContrast,
      [`gamegen-dashboard--theme-${theme}`]: theme !== 'system',
    },
    className
  );

  // Compute content classes
  const contentClasses = classnames(
    'gamegen-dashboard__content',
    {
      'gamegen-dashboard__content--collapsed': isSidebarCollapsed,
    }
  );

  return (
    <div 
      className={containerClasses}
      style={{
        ...styles.container,
        ...(gameMode.active && styles.gameMode),
      }}
      data-theme={theme}
      data-game-mode={gameMode.active}
      data-reduced-motion={reducedMotion || gameMode.reducedMotion}
      data-high-contrast={gameMode.highContrast}
    >
      <Header
        theme={{
          mode: theme,
          highContrast: gameMode.highContrast,
        }}
        performance={{
          reducedMotion: reducedMotion || gameMode.reducedMotion,
          lowLatency: gameMode.active,
        }}
      />

      <main style={styles.main}>
        <Sidebar
          isCollapsed={isSidebarCollapsed}
          onToggle={handleSidebarToggle}
          theme={theme === 'system' ? 
            (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') 
            : theme
          }
          highContrast={gameMode.highContrast}
        />

        <div 
          className={contentClasses}
          style={{
            ...styles.content,
            ...(isSidebarCollapsed && styles.contentCollapsed),
          }}
          role="main"
          aria-live={gameMode.active ? 'polite' : 'off'}
        >
          {children}
        </div>
      </main>
    </div>
  );
});

Dashboard.displayName = 'Dashboard';

export type { DashboardProps, GameMode };
export default Dashboard;