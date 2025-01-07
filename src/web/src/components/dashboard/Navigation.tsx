import React, { memo, useCallback, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom'; // @version ^6.0.0
import classnames from 'classnames'; // @version ^2.3.2
import { analytics } from '@segment/analytics-next'; // @version ^1.51.0

import { Icon } from '../common/Icon';
import { Button } from '../common/Button';
import { useAuth } from '../../hooks/useAuth';
import { Permission } from '../../types/auth';

// Navigation item interface with role-based access control
interface NavigationItem {
  path: string;
  label: string;
  icon: string;
  permission: Permission | null;
  analyticsId: string;
  shortcut: string;
}

// Navigation items with permissions and shortcuts
const NAVIGATION_ITEMS: NavigationItem[] = [
  {
    path: '/generate',
    label: 'Generate',
    icon: 'generate',
    permission: Permission.GENERATE_CONTENT,
    analyticsId: 'nav_generate',
    shortcut: 'Alt+G'
  },
  {
    path: '/control',
    label: 'Control',
    icon: 'control',
    permission: Permission.CONTROL_CONTENT,
    analyticsId: 'nav_control',
    shortcut: 'Alt+C'
  },
  {
    path: '/history',
    label: 'History',
    icon: 'history',
    permission: null,
    analyticsId: 'nav_history',
    shortcut: 'Alt+H'
  },
  {
    path: '/settings',
    label: 'Settings',
    icon: 'settings',
    permission: null,
    analyticsId: 'nav_settings',
    shortcut: 'Alt+S'
  }
];

const Navigation: React.FC = memo(() => {
  const navigate = useNavigate();
  const location = useLocation();
  const { checkPermission } = useAuth();

  // Filter navigation items based on user permissions
  const allowedItems = NAVIGATION_ITEMS.filter(item => 
    !item.permission || checkPermission(item.permission)
  );

  // Handle navigation with analytics tracking
  const handleNavigation = useCallback((item: NavigationItem) => {
    analytics.track('Navigation Click', {
      itemId: item.analyticsId,
      path: item.path,
      from: location.pathname
    });
    navigate(item.path);
  }, [navigate, location.pathname]);

  // Setup keyboard shortcuts
  useEffect(() => {
    const handleKeyboard = (event: KeyboardEvent) => {
      if (event.altKey) {
        const item = allowedItems.find(
          item => item.shortcut.toLowerCase() === `alt+${event.key.toLowerCase()}`
        );
        if (item) {
          event.preventDefault();
          handleNavigation(item);
        }
      }
    };

    window.addEventListener('keydown', handleKeyboard);
    return () => window.removeEventListener('keydown', handleKeyboard);
  }, [allowedItems, handleNavigation]);

  return (
    <nav
      className="gamegen-navigation"
      role="navigation"
      aria-label="Main navigation"
    >
      <div className="gamegen-navigation__container">
        {allowedItems.map((item) => (
          <Button
            key={item.path}
            variant="text"
            className={classnames('gamegen-navigation__item', {
              'gamegen-navigation__item--active': location.pathname === item.path
            })}
            onClick={() => handleNavigation(item)}
            icon={item.icon}
            iconPosition="left"
            aria-current={location.pathname === item.path ? 'page' : undefined}
            aria-label={`${item.label} (${item.shortcut})`}
            data-testid={`nav-${item.analyticsId}`}
          >
            <span className="gamegen-navigation__label">{item.label}</span>
            <span className="gamegen-navigation__shortcut" aria-hidden="true">
              {item.shortcut}
            </span>
          </Button>
        ))}
      </div>
    </nav>
  );
});

Navigation.displayName = 'Navigation';

// Styles for the navigation component
const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: 'var(--spacing-unit)',
    padding: 'var(--spacing-md)',
    backgroundColor: 'var(--background-color)',
    transition: 'var(--transition-speed)',
    '@media': {
      '(max-width: 768px)': {
        flexDirection: 'row',
        padding: 'var(--spacing-sm)'
      }
    }
  },
  button: {
    width: '100%',
    justifyContent: 'flex-start',
    padding: 'var(--spacing-md)',
    borderRadius: 'var(--border-radius)',
    transition: 'var(--transition-speed)',
    position: 'relative',
    overflow: 'hidden',
    '&:hover': {
      backgroundColor: 'var(--hover-color)',
      transform: 'scale(1.02)'
    },
    '&:focus': {
      outline: '2px solid var(--focus-color)',
      outlineOffset: '2px'
    }
  },
  active: {
    backgroundColor: 'var(--primary-color)',
    color: 'var(--text-color)',
    '&::after': {
      content: '""',
      position: 'absolute',
      inset: '0',
      background: 'var(--glow-gradient)',
      opacity: '0.2'
    }
  }
} as const;

export default Navigation;