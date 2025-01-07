import React, { useEffect, memo } from 'react'; // ^18.2.0
import { ErrorBoundary } from 'react-error-boundary'; // ^4.0.0
import { useTheme } from '../contexts/ThemeContext';
import { useAuth } from '../contexts/AuthContext';
import Loading from '../components/common/Loading';

// Interface for MainLayout props
interface MainLayoutProps {
  children: React.ReactNode;
}

// Error fallback component for production error handling
const ErrorFallback: React.FC<{ error: Error }> = ({ error }) => (
  <div
    role="alert"
    style={{
      padding: 'var(--spacing-md)',
      color: 'rgb(var(--error-rgb))',
      textAlign: 'center'
    }}
  >
    <h2>An error occurred in the application</h2>
    <pre style={{ fontSize: 'var(--font-size-sm)' }}>
      {process.env.NODE_ENV === 'development' ? error.message : 'Please try refreshing the page'}
    </pre>
  </div>
);

// Main layout component with theme and auth integration
const MainLayout: React.FC<MainLayoutProps> = memo(({ children }) => {
  const { theme, isDarkMode } = useTheme();
  const { isAuthenticated, checkAuth, authError } = useAuth();
  const [isLoading, setIsLoading] = React.useState(true);

  // Check authentication status on mount
  useEffect(() => {
    const initAuth = async () => {
      try {
        await checkAuth();
      } catch (error) {
        console.error('Auth check failed:', error);
      } finally {
        setIsLoading(false);
      }
    };

    initAuth();
  }, [checkAuth]);

  // Apply theme CSS variables to root element
  useEffect(() => {
    const root = document.documentElement;
    Object.entries(theme.colors).forEach(([key, value]) => {
      root.style.setProperty(`--${key}-rgb`, value.replace(/[^\d,]/g, ''));
    });
    root.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
  }, [theme, isDarkMode]);

  // Handle loading state with accessibility
  if (isLoading) {
    return (
      <div
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'rgb(var(--background-rgb))'
        }}
      >
        <Loading
          size="large"
          label="Loading application..."
          aria-label="Loading application content"
        />
      </div>
    );
  }

  // Handle authentication error
  if (authError) {
    return (
      <div
        role="alert"
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'rgb(var(--background-rgb))',
          color: 'rgb(var(--error-rgb))'
        }}
      >
        <div>Authentication error. Please try again later.</div>
      </div>
    );
  }

  return (
    <ErrorBoundary
      FallbackComponent={ErrorFallback}
      onError={(error) => {
        console.error('Layout error:', error);
        // Log to monitoring service in production
      }}
    >
      <div
        className="layout-container"
        style={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: 'rgb(var(--background-rgb))',
          color: 'rgb(var(--text-rgb))',
          transition: 'background-color var(--transition-speed-normal), color var(--transition-speed-normal)',
          WebkitFontSmoothing: 'antialiased',
          MozOsxFontSmoothing: 'grayscale'
        }}
      >
        <main
          className="layout-content"
          style={{
            flex: 1,
            width: '100%',
            maxWidth: theme.spacing.layout.container,
            margin: '0 auto',
            padding: 'var(--spacing-md)',
            display: 'grid',
            gridTemplateColumns: 'repeat(12, 1fr)',
            gap: 'var(--spacing-md)',
            '@media (max-width: 768px)': {
              gridTemplateColumns: 'repeat(4, 1fr)',
              padding: 'var(--spacing-sm)',
              gap: 'var(--spacing-sm)'
            }
          }}
        >
          {children}
        </main>
      </div>
    </ErrorBoundary>
  );
});

// Display name for debugging
MainLayout.displayName = 'MainLayout';

export default MainLayout;