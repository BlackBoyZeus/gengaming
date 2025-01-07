import { debounce } from 'lodash'; // v4.17.21

// Type definitions for control actions
export type ControlAction = 'FORWARD' | 'BACKWARD' | 'LEFT' | 'RIGHT' | 'ACTION';

// Immutable key mapping configuration
export const KEY_MAPPINGS: Readonly<Record<string, ControlAction>> = {
  'W': 'FORWARD',
  'S': 'BACKWARD',
  'A': 'LEFT',
  'D': 'RIGHT',
  'SPACE': 'ACTION'
} as const;

// Interface for control states
interface IControlStates {
  readonly FORWARD: boolean;
  readonly BACKWARD: boolean;
  readonly LEFT: boolean;
  readonly RIGHT: boolean;
  readonly ACTION: boolean;
}

// Initial control states
const CONTROL_STATES: IControlStates = {
  FORWARD: false,
  BACKWARD: false,
  LEFT: false,
  RIGHT: false,
  ACTION: false
};

// Performance optimization constants
const DEBOUNCE_THRESHOLD = 16; // 16ms for 60fps responsiveness
const keyCache = new Map<string, boolean>();

/**
 * Type-safe validation of keyboard keys against control mapping
 * @param key - Keyboard key to validate
 * @returns boolean indicating if key is mapped to a control
 */
export const isControlKey = (key: string): key is keyof typeof KEY_MAPPINGS => {
  if (!key || typeof key !== 'string') return false;
  
  const upperKey = key.toUpperCase();
  const cached = keyCache.get(upperKey);
  
  if (cached !== undefined) {
    return cached;
  }
  
  const isValid = Object.keys(KEY_MAPPINGS).includes(upperKey);
  keyCache.set(upperKey, isValid);
  return isValid;
};

/**
 * Maps keyboard keys to control actions with type safety
 * @param key - Keyboard key to map
 * @returns Mapped control action or null if not valid
 */
const getControlFromKey = (key: string): ControlAction | null => {
  if (!key || typeof key !== 'string') return null;
  
  const upperKey = key.toUpperCase();
  return isControlKey(upperKey) ? KEY_MAPPINGS[upperKey] : null;
};

/**
 * Creates an immutable copy of control states with updates
 * @param currentStates - Current control states
 * @param updates - Partial updates to apply
 * @returns New immutable control states
 */
const createUpdatedStates = (
  currentStates: IControlStates,
  updates: Partial<IControlStates>
): IControlStates => ({
  ...currentStates,
  ...updates
});

// Track active keys to prevent duplicate events
const activeKeys = new Set<string>();

/**
 * Debounced handler for keydown events
 * @param event - Keyboard event
 * @returns Updated control states
 */
export const handleKeyDown = debounce((event: KeyboardEvent): IControlStates => {
  // Prevent default only for mapped control keys
  if (isControlKey(event.key)) {
    event.preventDefault();
  }

  const key = event.code === 'Space' ? 'SPACE' : event.key;
  
  // Prevent duplicate keydown events
  if (activeKeys.has(key)) {
    return CONTROL_STATES;
  }
  
  const control = getControlFromKey(key);
  if (!control) {
    return CONTROL_STATES;
  }
  
  activeKeys.add(key);
  
  return createUpdatedStates(CONTROL_STATES, {
    [control]: true
  });
}, DEBOUNCE_THRESHOLD, { maxWait: DEBOUNCE_THRESHOLD * 2 });

/**
 * Debounced handler for keyup events
 * @param event - Keyboard event
 * @returns Updated control states
 */
export const handleKeyUp = debounce((event: KeyboardEvent): IControlStates => {
  const key = event.code === 'Space' ? 'SPACE' : event.key;
  
  // Clean up tracked keys
  activeKeys.delete(key);
  
  const control = getControlFromKey(key);
  if (!control) {
    return CONTROL_STATES;
  }
  
  return createUpdatedStates(CONTROL_STATES, {
    [control]: false
  });
}, DEBOUNCE_THRESHOLD, { maxWait: DEBOUNCE_THRESHOLD * 2 });

// Recovery function for stuck keys
const clearStuckKeys = debounce(() => {
  activeKeys.clear();
  Object.keys(CONTROL_STATES).forEach(key => {
    CONTROL_STATES[key as keyof IControlStates] = false;
  });
}, DEBOUNCE_THRESHOLD * 4);

// Initialize stuck key recovery on window blur
if (typeof window !== 'undefined') {
  window.addEventListener('blur', clearStuckKeys);
}