import React, { useCallback, useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { useTheme } from '@mui/material';
import { toast } from 'react-toastify';

import DashboardLayout from '../layouts/DashboardLayout';
import Button from '../components/common/Button';
import { useAuth } from '../hooks/useAuth';
import { Permission } from '../types/auth';
import { VIDEO_SETTINGS, UI_CONSTANTS, CONTROL_SETTINGS } from '../config/constants';

// Settings form validation schema
const settingsSchema = z.object({
  resolution: z.enum(['320x256', '848x480', '1280x720']),
  frameCount: z.number().min(24).max(102),
  perspective: z.enum(['first_person', 'third_person']),
  theme: z.enum(['light', 'dark', 'system']),
  performance: z.enum(['balanced', 'quality', 'speed']),
  shortcuts: z.record(z.string(), z.string())
});

type SettingsFormData = z.infer<typeof settingsSchema>;

// Default settings values
const defaultSettings: SettingsFormData = {
  resolution: '1280x720',
  frameCount: VIDEO_SETTINGS.DEFAULT_FRAME_COUNT,
  perspective: 'third_person',
  theme: 'system',
  performance: 'balanced',
  shortcuts: CONTROL_SETTINGS.CONTROL_MODES
};

const Settings: React.FC = () => {
  const { authState, checkPermission } = useAuth();
  const theme = useTheme();
  const [isLoading, setIsLoading] = useState(false);

  // Form initialization with validation
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isDirty }
  } = useForm<SettingsFormData>({
    resolver: zodResolver(settingsSchema),
    defaultValues: defaultSettings
  });

  // Load saved settings on mount
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const savedSettings = localStorage.getItem('gamegen_settings');
        if (savedSettings) {
          reset(JSON.parse(savedSettings));
        }
      } catch (error) {
        console.error('Failed to load settings:', error);
        toast.error('Failed to load settings');
      }
    };
    loadSettings();
  }, [reset]);

  // Handle settings submission
  const onSubmit = useCallback(async (data: SettingsFormData) => {
    setIsLoading(true);
    try {
      // Validate permissions for each setting type
      if (data.resolution !== defaultSettings.resolution) {
        await checkPermission(Permission.CONFIGURE_SYSTEM);
      }

      // Save settings
      localStorage.setItem('gamegen_settings', JSON.stringify(data));
      
      // Update theme if changed
      if (data.theme !== theme.palette.mode) {
        document.documentElement.dataset.theme = data.theme;
      }

      toast.success('Settings saved successfully');
    } catch (error) {
      console.error('Failed to save settings:', error);
      toast.error('Failed to save settings');
    } finally {
      setIsLoading(false);
    }
  }, [theme.palette.mode, checkPermission]);

  return (
    <DashboardLayout>
      <div className="settings-container" style={styles.container}>
        <h1 className="settings-title" style={styles.sectionTitle}>
          Settings
        </h1>

        <form onSubmit={handleSubmit(onSubmit)} aria-label="Settings form">
          {/* Video Generation Settings */}
          <section className="settings-section" style={styles.section}>
            <h2 style={styles.sectionTitle}>Video Generation</h2>
            
            <div style={styles.formGroup}>
              <label htmlFor="resolution" style={styles.label}>
                Resolution
              </label>
              <select
                id="resolution"
                {...register('resolution')}
                aria-invalid={!!errors.resolution}
                style={styles.input}
              >
                {VIDEO_SETTINGS.SUPPORTED_RESOLUTIONS.map(res => (
                  <option key={`${res.width}x${res.height}`} value={`${res.width}x${res.height}`}>
                    {`${res.width}x${res.height}`}
                  </option>
                ))}
              </select>
              {errors.resolution && (
                <span role="alert" style={styles.error}>
                  {errors.resolution.message}
                </span>
              )}
            </div>

            <div style={styles.formGroup}>
              <label htmlFor="frameCount" style={styles.label}>
                Frame Count
              </label>
              <input
                type="number"
                id="frameCount"
                {...register('frameCount', { valueAsNumber: true })}
                min={24}
                max={102}
                aria-invalid={!!errors.frameCount}
                style={styles.input}
              />
              {errors.frameCount && (
                <span role="alert" style={styles.error}>
                  {errors.frameCount.message}
                </span>
              )}
            </div>

            <div style={styles.formGroup}>
              <label htmlFor="perspective" style={styles.label}>
                Camera Perspective
              </label>
              <select
                id="perspective"
                {...register('perspective')}
                aria-invalid={!!errors.perspective}
                style={styles.input}
              >
                <option value="first_person">First Person</option>
                <option value="third_person">Third Person</option>
              </select>
            </div>
          </section>

          {/* Interface Settings */}
          <section className="settings-section" style={styles.section}>
            <h2 style={styles.sectionTitle}>Interface</h2>

            <div style={styles.formGroup}>
              <label htmlFor="theme" style={styles.label}>
                Theme
              </label>
              <select
                id="theme"
                {...register('theme')}
                aria-invalid={!!errors.theme}
                style={styles.input}
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="system">System</option>
              </select>
            </div>

            <div style={styles.formGroup}>
              <label htmlFor="performance" style={styles.label}>
                Performance Mode
              </label>
              <select
                id="performance"
                {...register('performance')}
                aria-invalid={!!errors.performance}
                style={styles.input}
              >
                <option value="balanced">Balanced</option>
                <option value="quality">Quality</option>
                <option value="speed">Speed</option>
              </select>
            </div>
          </section>

          {/* Action Buttons */}
          <div style={styles.actions}>
            <Button
              type="button"
              variant="outline"
              onClick={() => reset(defaultSettings)}
              disabled={!isDirty || isLoading}
              aria-label="Reset to defaults"
            >
              Reset
            </Button>
            <Button
              type="submit"
              variant="primary"
              loading={isLoading}
              disabled={!isDirty}
              aria-label="Save settings"
            >
              Save Changes
            </Button>
          </div>
        </form>
      </div>
    </DashboardLayout>
  );
};

// Styles object
const styles = {
  container: {
    maxWidth: '800px',
    margin: '0 auto',
    padding: 'var(--spacing-lg)',
    minHeight: '100vh'
  },
  section: {
    marginBottom: 'var(--spacing-xl)',
    backgroundColor: 'var(--background-secondary)',
    borderRadius: 'var(--border-radius)',
    padding: 'var(--spacing-lg)',
    boxShadow: 'var(--shadow-sm)'
  },
  sectionTitle: {
    fontSize: 'var(--font-size-lg)',
    fontWeight: '600',
    marginBottom: 'var(--spacing-md)',
    color: 'var(--text-primary)'
  },
  formGroup: {
    marginBottom: 'var(--spacing-md)',
    position: 'relative'
  },
  label: {
    display: 'block',
    marginBottom: 'var(--spacing-xs)',
    fontWeight: '500',
    color: 'var(--text-secondary)'
  },
  input: {
    width: '100%',
    padding: 'var(--spacing-sm)',
    borderRadius: 'var(--border-radius-sm)',
    border: '1px solid var(--border-color)',
    backgroundColor: 'var(--background-primary)',
    color: 'var(--text-primary)',
    fontSize: 'var(--font-size-md)'
  },
  error: {
    color: 'var(--error)',
    fontSize: 'var(--font-size-sm)',
    marginTop: 'var(--spacing-xs)'
  },
  actions: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: 'var(--spacing-md)',
    marginTop: 'var(--spacing-xl)',
    position: 'sticky',
    bottom: 'var(--spacing-md)'
  }
} as const;

export default Settings;