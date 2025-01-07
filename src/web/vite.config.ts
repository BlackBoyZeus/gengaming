import { defineConfig } from 'vite'; // ^4.0.0
import react from '@vitejs/plugin-react'; // ^4.0.0
import tsconfigPaths from 'vite-tsconfig-paths'; // ^4.0.0
import compression from 'vite-plugin-compression'; // ^0.5.1

// Type-safe environment variable interface
import type { ImportMetaEnv } from './src/vite-env.d.ts';

export default defineConfig(({ mode, command }) => {
  // Validate required environment variables
  const requiredEnvVars: Array<keyof ImportMetaEnv> = ['VITE_API_URL', 'VITE_WS_URL'];
  for (const envVar of requiredEnvVars) {
    if (!process.env[envVar]) {
      throw new Error(`Missing required environment variable: ${envVar}`);
    }
  }

  const isDev = mode === 'development';

  return {
    // Plugin configuration with conditional development options
    plugins: [
      react({
        fastRefresh: true,
        // Enable additional development features in dev mode
        babel: {
          plugins: isDev ? ['react-refresh/babel'] : []
        }
      }),
      tsconfigPaths(),
      // Enable Brotli compression in production
      !isDev && compression({
        algorithm: 'brotli',
        ext: '.br',
        threshold: 10240, // Only compress files > 10KB
        deleteOriginFile: false
      })
    ].filter(Boolean),

    // Development server configuration
    server: {
      port: 3000,
      host: true,
      cors: {
        origin: [
          'http://localhost:3000',
          process.env.VITE_API_URL
        ].filter(Boolean),
        methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        credentials: true
      },
      hmr: {
        overlay: true,
        clientPort: 3000,
        timeout: 120000 // Extended timeout for development
      },
      watch: {
        usePolling: true,
        interval: 1000
      }
    },

    // Production build configuration
    build: {
      target: ['chrome90', 'firefox88', 'safari14'],
      outDir: 'dist',
      sourcemap: true,
      minify: 'terser',
      terserOptions: {
        compress: {
          drop_console: !isDev,
          passes: 2,
          pure_getters: true,
          unsafe: true
        }
      },
      cssCodeSplit: true,
      rollupOptions: {
        output: {
          manualChunks: {
            // Core vendor chunk
            vendor: ['react', 'react-dom'],
            // Game-specific dependencies
            game: ['three', '@react-three/fiber'],
            // UI components
            ui: ['@mui/material', '@emotion/react']
          }
        }
      },
      // Ensure consistent chunk size
      chunkSizeWarningLimit: 1000,
      assetsInlineLimit: 4096 // 4KB
    },

    // Path resolution configuration
    resolve: {
      alias: {
        '@': '/src',
        '@components': '/src/components',
        '@hooks': '/src/hooks',
        '@utils': '/src/utils',
        '@assets': '/src/assets'
      }
    },

    // Dependency optimization
    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        'three',
        '@react-three/fiber'
      ],
      exclude: ['@ffmpeg/core'],
      esbuildOptions: {
        target: 'esnext',
        supported: {
          bigint: true
        }
      }
    },

    // Preview server configuration
    preview: {
      port: 3000,
      host: true,
      cors: true
    },

    // Environment variable handling
    envPrefix: 'VITE_',
    
    // Performance optimizations
    esbuild: {
      logOverride: { 'this-is-undefined-in-esm': 'silent' },
      legalComments: 'none',
      target: 'esnext'
    },

    // Cache configuration
    cacheDir: 'node_modules/.vite',

    // Worker thread configuration
    worker: {
      format: 'es',
      plugins: []
    }
  };
});