import React, { useCallback, useEffect, useState, memo } from 'react';
import classnames from 'classnames'; // @version ^2.3.2
import { Button } from '../common/Button';
import { Icon } from '../common/Icon';
import { useAuth } from '../../contexts/AuthContext';
import { Permission, UserRole } from '../../types/auth';
import { UI_CONSTANTS } from '../../config/constants';

// Performance monitoring decorator
const withPerformanceTracking = (WrappedComponent: React.ComponentType<HeaderProps>) => {
  return function WithPerformanceTracking(props: HeaderProps) {
    useEffect(() => {
      performance.mark('header-render-start');
      return () => {
        performance.mark('header-render-end');
        performance.measure('header-render', 'header-render-start', 'header-render-end');
      };
    }, []);
    return <WrappedComponent {...props} />;
  };
};

// Interface for theme configuration
export interface GameTheme {
  mode: 'light' | 'dark' | 'system';
  highContrast: boolean;
}

// Interface for performance mode settings
export interface PerformanceMode {
  reducedMotion: boolean;
  lowLatency: boolean;
}

// Props interface for the Header component
export interface HeaderProps {
  className?: string;
  theme?: GameTheme;
  performance?: PerformanceMode;
}

// Header component implementation
const HeaderBase: React.FC<HeaderProps> = ({
  className,
  theme = { mode: 'system', highContrast: false },
  performance = { reducedMotion: false, lowLatency: true }
}) => {
  const { user, logout, hasPermission } = useAuth();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [notifications, setNotifications] = useState<number>(0);

  // Setup keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === 'h') {
        setIsMenuOpen(prev => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, []);

  // Handle profile interaction with haptic feedback
  const handleProfileClick = useCallback(async () => {
    if (navigator.vibrate) {
      navigator.vibrate(50);
    }
    setIsMenuOpen(prev => !prev);
  }, []);

  // Handle secure logout
  const handleLogout = useCallback(async () => {
    try {
      await logout();
    } catch (error) {
      console.error('Logout failed:', error);
    }
  }, [logout]);

  // Compute header classes with performance optimizations
  const headerClasses = classnames(
    'gamegen-header',
    {
      'gamegen-header--high-contrast': theme.highContrast,
      'gamegen-header--reduced-motion': performance.reducedMotion,
      'gamegen-header--low-latency': performance.lowLatency,
      'gamegen-header--dark': theme.mode === 'dark',
      'gamegen-header--light': theme.mode === 'light'
    },
    className
  );

  // Base styles with hardware acceleration
  const styles: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 'var(--spacing-md)',
    backgroundColor: 'var(--background-color)',
    borderBottom: '1px solid var(--border-color)',
    height: '64px',
    transform: 'translateZ(0)',
    backfaceVisibility: 'hidden',
    willChange: 'transform',
    contain: 'layout style paint'
  };

  return (
    <header 
      className={headerClasses} 
      style={styles}
      role="banner"
      aria-label="Game dashboard header"
    >
      <div className="gamegen-header__logo">
        <Icon 
          name="generate" 
          size="lg"
          ariaLabel="GameGen-X Logo"
        />
        <span className="gamegen-header__title">GameGen-X</span>
      </div>

      <nav className="gamegen-header__controls">
        {user && (
          <>
            <Button
              variant="outline"
              size="md"
              icon="control"
              hapticFeedback
              highContrast={theme.highContrast}
              aria-label="Open game controls"
              onClick={() => hasPermission(Permission.CONTROL_CONTENT)}
            >
              Controls
            </Button>

            <Button
              variant="primary"
              size="md"
              icon="generate"
              hapticFeedback
              highContrast={theme.highContrast}
              aria-label="Generate new game content"
              onClick={() => hasPermission(Permission.GENERATE_CONTENT)}
            >
              Generate
            </Button>

            <div className="gamegen-header__user">
              <Button
                variant="text"
                size="md"
                hapticFeedback
                highContrast={theme.highContrast}
                aria-haspopup="menu"
                aria-expanded={isMenuOpen}
                onClick={handleProfileClick}
              >
                {user.email}
                {notifications > 0 && (
                  <span 
                    className="gamegen-header__notifications"
                    aria-label={`${notifications} notifications`}
                  >
                    {notifications}
                  </span>
                )}
              </Button>

              {isMenuOpen && (
                <div 
                  className="gamegen-header__menu"
                  role="menu"
                  aria-label="User menu"
                >
                  {user.role === UserRole.ADMIN && (
                    <Button
                      variant="text"
                      fullWidth
                      onClick={() => hasPermission(Permission.CONFIGURE_SYSTEM)}
                      role="menuitem"
                    >
                      Settings
                    </Button>
                  )}
                  <Button
                    variant="text"
                    fullWidth
                    onClick={handleLogout}
                    role="menuitem"
                  >
                    Logout
                  </Button>
                </div>
              )}
            </div>
          </>
        )}
      </nav>
    </header>
  );
};

// Apply performance tracking and memoization
const Header = memo(withPerformanceTracking(HeaderBase));
Header.displayName = 'Header';

export default Header;
export type { HeaderProps, GameTheme, PerformanceMode };