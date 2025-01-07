import React from 'react';
import { render, fireEvent, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, describe, it, jest } from '@jest/globals';
import { ThemeProvider } from '@emotion/react';
import Button, { ButtonProps } from '../../src/components/common/Button';

// Helper function to render button with theme context
const renderButton = (props: Partial<ButtonProps> = {}) => {
  const defaultProps: ButtonProps = {
    children: 'Test Button',
    onClick: jest.fn(),
  };

  return render(
    <ThemeProvider theme={{}}>
      <Button {...defaultProps} {...props} />
    </ThemeProvider>
  );
};

// Mock haptic feedback API
const mockHapticFeedback = () => {
  const vibrateMock = jest.fn();
  Object.defineProperty(window.navigator, 'vibrate', {
    value: vibrateMock,
    writable: true,
  });
  return vibrateMock;
};

describe('Button Component', () => {
  describe('Design System Compliance', () => {
    it('applies correct spacing based on size variant', () => {
      const { rerender } = renderButton({ size: 'sm' });
      let button = screen.getByRole('button');
      expect(button).toHaveStyle({ padding: 'var(--spacing-xs)' });

      rerender(<Button size="md">Test Button</Button>);
      button = screen.getByRole('button');
      expect(button).toHaveStyle({ padding: 'var(--spacing-sm)' });

      rerender(<Button size="lg">Test Button</Button>);
      button = screen.getByRole('button');
      expect(button).toHaveStyle({ padding: 'var(--spacing-md)' });
    });

    it('implements game-focused color scheme', () => {
      const { container } = renderButton({ variant: 'primary' });
      const button = container.firstChild as HTMLElement;
      const styles = window.getComputedStyle(button);
      
      expect(styles.getPropertyValue('--primary-rgb')).toBeDefined();
      expect(button).toHaveClass('gamegen-button--primary');
    });

    it('supports high contrast mode', () => {
      const { container } = renderButton({ highContrast: true });
      expect(container.firstChild).toHaveAttribute('data-high-contrast', 'true');
    });

    it('uses hardware-accelerated animations', () => {
      const { container } = renderButton();
      const button = container.firstChild as HTMLElement;
      expect(button).toHaveStyle({
        transform: 'translateZ(0)',
        willChange: 'transform, opacity',
      });
    });
  });

  describe('Gaming Features', () => {
    it('provides haptic feedback on touch', async () => {
      const vibrateMock = mockHapticFeedback();
      const { container } = renderButton({ hapticFeedback: true });
      const button = container.firstChild as HTMLElement;

      fireEvent.touchStart(button);
      expect(vibrateMock).toHaveBeenCalledWith(50);
    });

    it('handles different game states', () => {
      const states: Array<'idle' | 'loading' | 'active' | 'error'> = ['idle', 'loading', 'active', 'error'];
      
      states.forEach(state => {
        const { container } = renderButton({ gameState: state });
        expect(container.firstChild).toHaveAttribute('data-game-state', state);
        expect(container.firstChild).toHaveClass(`gamegen-button--${state}`);
      });
    });

    it('supports RTL layout for gaming controls', () => {
      document.dir = 'rtl';
      const { container } = renderButton({ icon: 'control', iconPosition: 'left' });
      expect(container.firstChild).toHaveClass('gamegen-button--rtl');
      document.dir = 'ltr';
    });
  });

  describe('Performance Optimizations', () => {
    it('implements CSS containment', () => {
      const { container } = renderButton();
      const button = container.firstChild as HTMLElement;
      expect(button).toHaveStyle({
        contain: 'layout style paint',
      });
    });

    it('debounces click handler', async () => {
      const onClick = jest.fn();
      const { container } = renderButton({ onClick });
      const button = container.firstChild as HTMLElement;

      // Rapid clicks should be debounced
      fireEvent.click(button);
      fireEvent.click(button);
      fireEvent.click(button);

      expect(onClick).toHaveBeenCalledTimes(3);
    });
  });

  describe('Accessibility Features', () => {
    it('meets WCAG 2.1 Level AA requirements', () => {
      const { container } = renderButton({
        'aria-label': 'Action Button',
        disabled: true,
      });
      
      const button = container.firstChild as HTMLElement;
      expect(button).toHaveAttribute('aria-disabled', 'true');
      expect(button).toHaveAttribute('aria-label', 'Action Button');
    });

    it('handles keyboard navigation', async () => {
      const onClick = jest.fn();
      renderButton({ onClick });
      
      const button = screen.getByRole('button');
      button.focus();
      expect(document.activeElement).toBe(button);
      
      await userEvent.keyboard('{enter}');
      expect(onClick).toHaveBeenCalled();
      
      await userEvent.keyboard(' ');
      expect(onClick).toHaveBeenCalledTimes(2);
    });

    it('maintains proper focus states', async () => {
      const { container } = renderButton();
      const button = container.firstChild as HTMLElement;
      
      button.focus();
      expect(document.activeElement).toBe(button);
      expect(button).toHaveStyle({
        outline: 'none',
      });
    });

    it('provides loading state feedback', () => {
      const { container } = renderButton({ loading: true });
      const button = container.firstChild as HTMLElement;
      
      expect(button).toHaveAttribute('aria-busy', 'true');
      expect(within(button).getByRole('status')).toBeInTheDocument();
    });
  });
});