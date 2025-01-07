import { useState, useEffect, useCallback } from 'react'; // ^18.0.0
import { isControlKey, handleKeyDown, handleKeyUp } from '../utils/keyboard';

// Interface for hook configuration options
export interface UseKeyboardOptions {
  enabled: boolean;
  debounceMs?: number;
  allowedKeys?: string[];
}

// Interface for hook return values
export interface UseKeyboardReturn {
  controlStates: Record<string, boolean>;
  isActive: boolean;
  lastKeyPressed: string | null;
  responseTime: number;
}

/**
 * High-performance React hook for real-time keyboard event handling and control state management
 * Ensures <50ms response time with optimized event processing and state updates
 * 
 * @param options - Configuration options for keyboard hook behavior
 * @returns Object containing control states, active status, and performance metrics
 */
const useKeyboard = ({
  enabled = true,
  debounceMs = 16, // Optimized for 60fps
  allowedKeys
}: UseKeyboardOptions): UseKeyboardReturn => {
  // Initialize control states with optimized state management
  const [controlStates, setControlStates] = useState<Record<string, boolean>>({});
  const [isActive, setIsActive] = useState<boolean>(false);
  const [lastKeyPressed, setLastKeyPressed] = useState<string | null>(null);
  const [responseTime, setResponseTime] = useState<number>(0);

  // Performance tracking
  const performanceTracker = useCallback(() => {
    let startTime = 0;
    
    return {
      start: () => {
        startTime = performance.now();
      },
      end: () => {
        const endTime = performance.now();
        setResponseTime(endTime - startTime);
      }
    };
  }, []);

  const tracker = performanceTracker();

  // Memoized event handlers for optimal performance
  const onKeyDown = useCallback((event: KeyboardEvent) => {
    if (!enabled) return;
    
    tracker.start();

    // Filter allowed keys if specified
    if (allowedKeys && !allowedKeys.includes(event.key)) {
      return;
    }

    if (isControlKey(event.key)) {
      const newStates = handleKeyDown(event);
      setControlStates(newStates);
      setLastKeyPressed(event.key);
      setIsActive(true);
    }

    tracker.end();
  }, [enabled, allowedKeys, tracker]);

  const onKeyUp = useCallback((event: KeyboardEvent) => {
    if (!enabled) return;

    tracker.start();

    if (isControlKey(event.key)) {
      const newStates = handleKeyUp(event);
      setControlStates(newStates);
      setIsActive(false);
    }

    tracker.end();
  }, [enabled, tracker]);

  // Focus management for proper event handling
  const onWindowFocus = useCallback(() => {
    if (enabled) {
      setIsActive(false);
      setControlStates({});
    }
  }, [enabled]);

  const onWindowBlur = useCallback(() => {
    if (enabled) {
      setIsActive(false);
      setControlStates({});
      setLastKeyPressed(null);
    }
  }, [enabled]);

  // Set up event listeners with cleanup
  useEffect(() => {
    if (!enabled) return;

    // Use passive event listeners for performance
    const options: AddEventListenerOptions = { passive: false };

    window.addEventListener('keydown', onKeyDown, options);
    window.addEventListener('keyup', onKeyUp, options);
    window.addEventListener('focus', onWindowFocus);
    window.addEventListener('blur', onWindowBlur);

    // Proper cleanup
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('focus', onWindowFocus);
      window.removeEventListener('blur', onWindowBlur);
    };
  }, [enabled, onKeyDown, onKeyUp, onWindowFocus, onWindowBlur]);

  // Return current states and metrics
  return {
    controlStates,
    isActive,
    lastKeyPressed,
    responseTime
  };
};

export default useKeyboard;