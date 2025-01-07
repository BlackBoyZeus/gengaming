import React, { useCallback, useRef } from 'react';
import styled from '@emotion/styled';
import FocusTrap from 'focus-trap-react';
import { Theme } from '../../config/theme';

// Keyframes for animations
const fadeIn = `
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
`;

const slideIn = `
  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translate3d(0,-20px,0);
    }
    to {
      opacity: 1;
      transform: translate3d(0,0,0);
    }
  }
`;

// Styled components with GPU acceleration and RTL support
const ModalOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: ${(props: { theme: Theme }) => props.theme.colors.overlay};
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  transform: translate3d(0,0,0);
  will-change: opacity;
  animation: fadeIn 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  ${fadeIn}
`;

const ModalContent = styled.div<{ width?: string; height?: string }>`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.effects.borderRadius.md};
  padding: ${props => props.theme.spacing.scale.lg};
  width: ${props => props.width || 'auto'};
  height: ${props => props.height || 'auto'};
  max-width: 90vw;
  max-height: 90vh;
  overflow: auto;
  position: relative;
  will-change: transform, opacity;
  transform: translate3d(0,0,0);
  animation: slideIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  direction: inherit;
  box-shadow: ${props => props.theme.effects.shadows.lg};
  ${slideIn}
`;

const ModalHeader = styled.header`
  margin-bottom: ${props => props.theme.spacing.scale.md};
`;

const ModalTitle = styled.h2`
  margin: 0;
  color: ${props => props.theme.colors.text};
  font-size: ${props => props.theme.typography.sizeScale.xl};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
`;

const CloseButton = styled.button`
  position: absolute;
  top: ${props => props.theme.spacing.scale.md};
  right: ${props => props.theme.spacing.scale.md};
  background: transparent;
  border: none;
  color: ${props => props.theme.colors.textSecondary};
  cursor: pointer;
  padding: ${props => props.theme.spacing.scale.xs};
  border-radius: ${props => props.theme.effects.borderRadius.sm};
  transition: color ${props => props.theme.effects.transitions.fast};

  &:hover {
    color: ${props => props.theme.colors.text};
  }

  &:focus {
    outline: none;
    box-shadow: 0 0 0 2px ${props => props.theme.colors.focus};
  }
`;

// Interface for Modal props
interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  width?: string;
  height?: string;
  ariaLabel?: string;
  initialFocusRef?: React.RefObject<HTMLElement>;
}

// Custom hook for modal event handlers
const useModalHandlers = (
  onClose: () => void,
  overlayRef: React.RefObject<HTMLDivElement>
) => {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    },
    [onClose]
  );

  const handleOverlayClick = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (event.target === overlayRef.current) {
        onClose();
      }
    },
    [onClose, overlayRef]
  );

  return {
    handleKeyDown,
    handleOverlayClick
  };
};

// Memoized Modal component
export const Modal = React.memo<ModalProps>(({
  isOpen,
  onClose,
  title,
  children,
  width,
  height,
  ariaLabel,
  initialFocusRef
}) => {
  const overlayRef = useRef<HTMLDivElement>(null);
  const { handleKeyDown, handleOverlayClick } = useModalHandlers(onClose, overlayRef);

  // Don't render if modal is not open
  if (!isOpen) return null;

  return (
    <FocusTrap
      focusTrapOptions={{
        initialFocus: initialFocusRef,
        escapeDeactivates: true,
        allowOutsideClick: true
      }}
    >
      <ModalOverlay
        ref={overlayRef}
        onClick={handleOverlayClick}
        onKeyDown={handleKeyDown}
        role="dialog"
        aria-modal="true"
        aria-label={ariaLabel || title}
      >
        <ModalContent
          width={width}
          height={height}
          role="document"
        >
          <ModalHeader>
            <ModalTitle>{title}</ModalTitle>
            <CloseButton
              onClick={onClose}
              aria-label="Close modal"
              type="button"
            >
              ✕
            </CloseButton>
          </ModalHeader>
          {children}
        </ModalContent>
      </ModalOverlay>
    </FocusTrap>
  );
});

Modal.displayName = 'Modal';

export default Modal;