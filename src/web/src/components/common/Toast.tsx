import React, { useEffect, useState, useCallback } from 'react'; // ^18.0.0
import styled, { keyframes } from 'styled-components'; // ^5.3.0
import useSound from 'use-sound'; // ^4.0.1
import { useTheme } from '../../contexts/ThemeContext';

// Toast animation keyframes
const slideIn = keyframes`
  0% { transform: translateX(100%); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
`;

const slideOut = keyframes`
  0% { transform: translateX(0); opacity: 1; }
  100% { transform: translateX(100%); opacity: 0; }
`;

// Toast container with gaming aesthetics
const ToastContainer = styled.div<{
  position: string;
  type: string;
  isExiting: boolean;
  backgroundColor: string;
  borderColor: string;
}>`
  position: fixed;
  z-index: 1000;
  min-width: 300px;
  padding: ${({ theme }) => theme.spacing.scale.md};
  border-radius: ${({ theme }) => theme.effects.borderRadius.md};
  box-shadow: ${({ theme }) => theme.effects.shadows.lg};
  backdrop-filter: blur(8px);
  animation: ${({ isExiting }) => (isExiting ? slideOut : slideIn)} 0.3s ease-in-out;
  border: 2px solid ${({ borderColor }) => borderColor};
  background: ${({ backgroundColor }) => backgroundColor};
  transition: all 0.3s ease-in-out;
  
  ${({ position }) => {
    switch (position) {
      case 'top':
        return 'top: 20px; left: 50%; transform: translateX(-50%);';
      case 'bottom':
        return 'bottom: 20px; left: 50%; transform: translateX(-50%);';
      case 'top-left':
        return 'top: 20px; left: 20px;';
      case 'top-right':
        return 'top: 20px; right: 20px;';
      case 'bottom-left':
        return 'bottom: 20px; left: 20px;';
      case 'bottom-right':
        return 'bottom: 20px; right: 20px;';
      default:
        return 'top: 20px; right: 20px;';
    }
  }}
`;

const ToastMessage = styled.p`
  margin: 0;
  font-family: ${({ theme }) => theme.typography.fontFamily};
  font-size: ${({ theme }) => theme.typography.sizeScale.sm};
  line-height: ${({ theme }) => theme.typography.lineHeight.normal};
  color: ${({ theme }) => theme.colors.text};
  text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
`;

// Toast props interface
interface ToastProps {
  message: string;
  type: 'success' | 'error' | 'warning' | 'game-event';
  duration?: number;
  onClose?: () => void;
  soundEnabled?: boolean;
  position?: 'top' | 'bottom' | 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
}

// Get toast styles based on type
const getToastStyles = (type: string, theme: any) => {
  const styles = {
    success: {
      background: `${theme.colors.success}33`,
      border: theme.colors.success,
      glow: theme.effects.glows.success
    },
    error: {
      background: `${theme.colors.error}33`,
      border: theme.colors.error,
      glow: theme.effects.glows.error
    },
    warning: {
      background: `${theme.colors.warning}33`,
      border: theme.colors.warning,
      glow: '0 0 15px rgba(241, 196, 15, 0.6)'
    },
    'game-event': {
      background: `${theme.colors.primary}33`,
      border: theme.colors.primary,
      glow: theme.effects.glows.primary
    }
  };

  return styles[type] || styles['game-event'];
};

// Sound effect mapping
const SOUND_EFFECTS = {
  success: '/sounds/success.mp3',
  error: '/sounds/error.mp3',
  warning: '/sounds/warning.mp3',
  'game-event': '/sounds/event.mp3'
};

export const Toast: React.FC<ToastProps> = ({
  message,
  type,
  duration = 3000,
  onClose,
  soundEnabled = true,
  position = 'top-right'
}) => {
  const { theme } = useTheme();
  const [isVisible, setIsVisible] = useState(true);
  const [isExiting, setIsExiting] = useState(false);
  const styles = getToastStyles(type, theme);
  
  // Initialize sound effect
  const [playSound] = useSound(SOUND_EFFECTS[type], {
    volume: 0.5,
    interrupt: true
  });

  const handleClose = useCallback(() => {
    setIsExiting(true);
    setTimeout(() => {
      setIsVisible(false);
      onClose?.();
    }, 300);
  }, [onClose]);

  // Auto-dismiss timer
  useEffect(() => {
    if (duration) {
      const timer = setTimeout(handleClose, duration);
      return () => clearTimeout(timer);
    }
  }, [duration, handleClose]);

  // Play sound effect on mount
  useEffect(() => {
    if (soundEnabled) {
      playSound();
    }
  }, [soundEnabled, playSound]);

  if (!isVisible) return null;

  return (
    <ToastContainer
      role="alert"
      aria-live="polite"
      position={position}
      type={type}
      isExiting={isExiting}
      backgroundColor={styles.background}
      borderColor={styles.border}
      style={{ boxShadow: styles.glow }}
    >
      <ToastMessage>{message}</ToastMessage>
    </ToastContainer>
  );
};

export default Toast;