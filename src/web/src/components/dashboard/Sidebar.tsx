import React, { memo, useCallback, useRef, useEffect } from 'react';
import classnames from 'classnames'; // @version ^2.3.2
import { makeStyles } from '@mui/styles'; // @version ^5.0.0

import Navigation from './Navigation';
import { Icon } from '../common/Icon';

// Interface for Sidebar component props
interface SidebarProps {
  className?: string;
  isCollapsed: boolean;
  onToggle: () => void;
  theme: 'light' | 'dark';
  highContrast: boolean;
}

// Styles using MUI's makeStyles for optimal performance
const useStyles = makeStyles(() => ({
  sidebar: {
    position: 'fixed',
    top: '64px',
    left: '0',
    bottom: '0',
    width: '280px',
    backgroundColor: 'var(--background-color)',
    borderRight: '1px solid var(--border-color)',
    transition: 'transform var(--transition-speed-normal) var(--transition-timing)',
    transform: 'translateZ(0)',
    willChange: 'transform',
    contain: 'layout size',
    zIndex: 'var(--z-index-sidebar)',
    '@media (prefers-reduced-motion: reduce)': {
      transition: 'none'
    },
    '@media (max-width: 768px)': {
      width: '100%',
      transform: 'translateX(-100%) translateZ(0)'
    }
  },
  collapsed: {
    width: '64px',
    transform: 'translateX(0) translateZ(0)',
    '@media (max-width: 768px)': {
      transform: 'translateX(-64px) translateZ(0)'
    }
  },
  content: {
    height: '100%',
    overflow: 'auto',
    padding: 'var(--spacing-md)',
    scrollbarGutter: 'stable',
    contain: 'paint',
    '&::-webkit-scrollbar': {
      width: '8px'
    },
    '&::-webkit-scrollbar-track': {
      background: 'transparent'
    },
    '&::-webkit-scrollbar-thumb': {
      background: 'var(--border-color)',
      borderRadius: '4px'
    }
  },
  toggle: {
    position: 'absolute',
    top: 'var(--spacing-md)',
    right: '-12px',
    zIndex: 'var(--z-index-sidebar-toggle)',
    transform: 'translateZ(0)',
    willChange: 'transform',
    backgroundColor: 'var(--background-color)',
    border: '1px solid var(--border-color)',
    borderRadius: '50%',
    padding: '4px',
    cursor: 'pointer',
    transition: 'background-color var(--transition-speed-fast)',
    '&:hover': {
      backgroundColor: 'var(--hover-color)'
    },
    '&:focus': {
      outline: 'none',
      boxShadow: '0 0 0 2px var(--focus-color)'
    }
  },
  highContrast: {
    borderRight: '2px solid var(--high-contrast-border)',
    backgroundColor: 'var(--high-contrast-background)',
    '& $toggle': {
      border: '2px solid var(--high-contrast-border)',
      backgroundColor: 'var(--high-contrast-background)'
    }
  }
}));

const Sidebar = memo(({
  className,
  isCollapsed,
  onToggle,
  theme,
  highContrast
}: SidebarProps) => {
  const classes = useStyles();
  const sidebarRef = useRef<HTMLDivElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);

  // Handle keyboard navigation within sidebar
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (event.key === 'Escape' && !isCollapsed) {
      onToggle();
      toggleRef.current?.focus();
    }
  }, [isCollapsed, onToggle]);

  // Setup keyboard event listeners
  useEffect(() => {
    const sidebar = sidebarRef.current;
    if (sidebar) {
      sidebar.addEventListener('keydown', handleKeyDown);
      return () => sidebar.removeEventListener('keydown', handleKeyDown);
    }
  }, [handleKeyDown]);

  // Debounced toggle handler with animation frame optimization
  const handleToggle = useCallback(() => {
    requestAnimationFrame(() => {
      onToggle();
    });
  }, [onToggle]);

  return (
    <aside
      ref={sidebarRef}
      className={classnames(
        classes.sidebar,
        {
          [classes.collapsed]: isCollapsed,
          [classes.highContrast]: highContrast
        },
        className
      )}
      data-theme={theme}
      data-high-contrast={highContrast}
      aria-expanded={!isCollapsed}
      aria-label="Main sidebar"
      role="complementary"
    >
      <button
        ref={toggleRef}
        className={classes.toggle}
        onClick={handleToggle}
        aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        aria-controls="sidebar-content"
        aria-expanded={!isCollapsed}
      >
        <Icon
          name="control"
          size="sm"
          ariaLabel=""
          style={{
            transform: `rotate(${isCollapsed ? 0 : 180}deg)`,
            transition: 'transform var(--transition-speed-fast)'
          }}
        />
      </button>

      <div
        id="sidebar-content"
        className={classes.content}
        role="navigation"
        aria-label="Main navigation"
      >
        <Navigation />
      </div>
    </aside>
  );
});

Sidebar.displayName = 'Sidebar';

export type { SidebarProps };
export default Sidebar;