import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useErrorBoundary } from 'react-error-boundary';
import { useAuth } from '../contexts/AuthContext';
import MainLayout from '../layouts/MainLayout';
import Dashboard from '../components/dashboard/Dashboard';
import GenerationForm from '../components/generation/GenerationForm';
import { GenerationMetrics } from '../types/generation';
import { UI_CONSTANTS, CONTROL_SETTINGS } from '../config/constants';

/**
 * Home page component for GameGen-X
 * Implements secure video generation interface with real-time interaction
 */
const Home: React.FC = React.memo(() => {
  const navigate = useNavigate();
  const { showBoundary } = useErrorBoundary();
  const { authState, validateSession } = useAuth();
  const [gameMode, setGameMode] = useState({
    active: false,
    reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    highContrast: window.matchMedia('(prefers-contrast: more)').matches
  });

  // Validate authentication on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const isValid = await validateSession();
        if (!isValid) {
          navigate('/login');
        }
      } catch (error) {
        showBoundary(error);
      }
    };
    checkAuth();
  }, [validateSession, navigate, showBoundary]);

  // Monitor system preferences
  useEffect(() => {
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const contrastQuery = window.matchMedia('(prefers-contrast: more)');

    const handleMotionChange = (e: MediaQueryListEvent) => {
      setGameMode(prev => ({ ...prev, reducedMotion: e.matches }));
    };

    const handleContrastChange = (e: MediaQueryListEvent) => {
      setGameMode(prev => ({ ...prev, highContrast: e.matches }));
    };

    motionQuery.addEventListener('change', handleMotionChange);
    contrastQuery.addEventListener('change', handleContrastChange);

    return () => {
      motionQuery.removeEventListener('change', handleMotionChange);
      contrastQuery.removeEventListener('change', handleContrastChange);
    };
  }, []);

  // Handle generation start with debouncing
  const handleGenerationStart = useCallback((generationId: string, metrics: GenerationMetrics) => {
    let timeoutId: NodeJS.Timeout;
    return () => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        setGameMode(prev => ({ ...prev, active: true }));
        navigate(`/control/${generationId}`, { 
          state: { metrics, timestamp: Date.now() }
        });
      }, CONTROL_SETTINGS.DEBOUNCE_DELAY);
    };
  }, [navigate]);

  // Handle generation completion
  const handleGenerationComplete = useCallback((generationId: string, metrics: GenerationMetrics) => {
    setGameMode(prev => ({ ...prev, active: false }));
    navigate(`/history/${generationId}`, {
      state: { metrics, timestamp: Date.now() }
    });
  }, [navigate]);

  // Handle generation errors
  const handleGenerationError = useCallback((error: Error) => {
    showBoundary(error);
    setGameMode(prev => ({ ...prev, active: false }));
  }, [showBoundary]);

  if (!authState.isAuthenticated) {
    return null;
  }

  return (
    <MainLayout>
      <Dashboard
        gameMode={gameMode}
        reducedMotion={gameMode.reducedMotion}
      >
        <div className="home-container">
          <h1 className="home-title">
            GameGen-X Video Generation
          </h1>
          <p className="home-description">
            Generate and control real-time game video content with AI-powered technology
          </p>
          <GenerationForm
            onGenerationStart={handleGenerationStart}
            onGenerationComplete={handleGenerationComplete}
            onError={handleGenerationError}
            className="home-generation-form"
            highContrastMode={gameMode.highContrast}
          />
        </div>
      </Dashboard>

      <style jsx>{`
        .home-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: var(--spacing-lg);
          width: 100%;
          max-width: ${UI_CONSTANTS.VIDEO_SETTINGS.DEFAULT_RESOLUTION.width}px;
          margin: 0 auto;
          min-height: 100vh;
        }

        .home-title {
          font-size: var(--font-size-2xl);
          font-weight: bold;
          margin-bottom: var(--spacing-lg);
          text-align: center;
          color: rgb(var(--text-rgb));
        }

        .home-description {
          font-size: var(--font-size-lg);
          margin-bottom: var(--spacing-xl);
          text-align: center;
          max-width: 800px;
          color: rgb(var(--text-rgb));
        }

        .home-generation-form {
          width: 100%;
          max-width: 800px;
        }

        @media (prefers-reduced-motion: reduce) {
          * {
            animation: none !important;
            transition: none !important;
          }
        }

        @media (max-width: 768px) {
          .home-container {
            padding: var(--spacing-md);
          }

          .home-title {
            font-size: var(--font-size-xl);
          }

          .home-description {
            font-size: var(--font-size-md);
          }
        }
      `}</style>
    </MainLayout>
  );
});

Home.displayName = 'Home';

export default Home;