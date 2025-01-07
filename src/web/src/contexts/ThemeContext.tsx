import { createContext, useContext, useState, useEffect, useCallback, useMemo, ReactNode } from 'react'; // ^18.0.0
import { Theme, lightTheme, darkTheme } from 'config/theme';
import { getItem, setItem } from 'utils/storage';

// Storage keys for theme preferences
const THEME_STORAGE_KEY = 'theme_preference';
const SYSTEM_THEME_KEY = 'use_system_theme';
const THEME_TRANSITION_MS = 150;

// Theme context interface
interface ThemeContextType {
  theme: Theme;
  isDarkMode: boolean;
  isSystemTheme: boolean;
  toggleTheme: () => void;
  setSystemTheme: (enabled: boolean) => void;
}

// Props interface for ThemeProvider
interface ThemeProviderProps {
  children: ReactNode;
  defaultTheme?: Theme;
  useSystemTheme?: boolean;
}

// Create theme context with null default value
const ThemeContext = createContext<ThemeContextType | null>(null);

// Detect system color scheme preference
const getSystemTheme = (): boolean => {
  if (typeof window === 'undefined') return false;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false;
};

// Theme provider component
export const ThemeProvider = ({
  children,
  defaultTheme,
  useSystemTheme = true,
}: ThemeProviderProps) => {
  // Theme state management
  const [theme, setTheme] = useState<Theme>(defaultTheme || lightTheme);
  const [isDarkMode, setIsDarkMode] = useState<boolean>(theme.name === 'dark');
  const [isSystemTheme, setIsSystemTheme] = useState<boolean>(useSystemTheme);

  // Initialize theme preferences from storage
  useEffect(() => {
    const initializeTheme = async () => {
      try {
        const storedSystemTheme = await getItem(SYSTEM_THEME_KEY);
        const useStoredSystemTheme = storedSystemTheme !== null ? storedSystemTheme : useSystemTheme;
        setIsSystemTheme(useStoredSystemTheme);

        if (useStoredSystemTheme) {
          setIsDarkMode(getSystemTheme());
          setTheme(getSystemTheme() ? darkTheme : lightTheme);
        } else {
          const storedTheme = await getItem(THEME_STORAGE_KEY);
          if (storedTheme === 'dark') {
            setIsDarkMode(true);
            setTheme(darkTheme);
          } else if (storedTheme === 'light') {
            setIsDarkMode(false);
            setTheme(lightTheme);
          }
        }
      } catch (error) {
        console.error('Failed to load theme preferences:', error);
      }
    };

    initializeTheme();
  }, [useSystemTheme]);

  // Listen for system theme changes
  useEffect(() => {
    if (!isSystemTheme) return;

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e: MediaQueryListEvent) => {
      setIsDarkMode(e.matches);
      setTheme(e.matches ? darkTheme : lightTheme);
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [isSystemTheme]);

  // Apply theme transitions
  useEffect(() => {
    document.documentElement.style.setProperty(
      'transition',
      `background-color ${THEME_TRANSITION_MS}ms ease-in-out, color ${THEME_TRANSITION_MS}ms ease-in-out`
    );

    return () => {
      document.documentElement.style.removeProperty('transition');
    };
  }, []);

  // Toggle theme handler
  const toggleTheme = useCallback(async () => {
    if (isSystemTheme) {
      await setItem(SYSTEM_THEME_KEY, false);
      setIsSystemTheme(false);
    }

    const newIsDarkMode = !isDarkMode;
    setIsDarkMode(newIsDarkMode);
    setTheme(newIsDarkMode ? darkTheme : lightTheme);
    await setItem(THEME_STORAGE_KEY, newIsDarkMode ? 'dark' : 'light');
  }, [isDarkMode, isSystemTheme]);

  // System theme handler
  const setSystemTheme = useCallback(async (enabled: boolean) => {
    setIsSystemTheme(enabled);
    await setItem(SYSTEM_THEME_KEY, enabled);

    if (enabled) {
      const systemDark = getSystemTheme();
      setIsDarkMode(systemDark);
      setTheme(systemDark ? darkTheme : lightTheme);
    }
  }, []);

  // Memoized context value
  const contextValue = useMemo(
    () => ({
      theme,
      isDarkMode,
      isSystemTheme,
      toggleTheme,
      setSystemTheme,
    }),
    [theme, isDarkMode, isSystemTheme, toggleTheme, setSystemTheme]
  );

  return (
    <ThemeContext.Provider value={contextValue}>
      {children}
    </ThemeContext.Provider>
  );
};

// Custom hook for accessing theme context
export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};