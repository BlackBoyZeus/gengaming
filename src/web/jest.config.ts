import type { Config } from 'jest';
import { pathsToModuleNameMapper } from 'ts-jest';

// Import tsconfig for path mapping
const tsconfig = require('./tsconfig.json');

const config: Config = {
  // Use jsdom environment for browser API simulation
  // @jest-environment-jsdom v29.5.0
  testEnvironment: 'jsdom',

  // Setup files for browser compatibility and performance testing
  setupFilesAfterEnv: [
    '<rootDir>/tests/setup.ts'
  ],

  // Module resolution and path mapping
  moduleNameMapper: {
    // Map TypeScript paths from tsconfig
    ...pathsToModuleNameMapper(tsconfig.compilerOptions.paths || {}, {
      prefix: '<rootDir>/'
    }),
    // Handle static assets
    '\\.(jpg|jpeg|png|gif|webp|svg)$': '<rootDir>/tests/__mocks__/fileMock.ts',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    // Handle web workers
    '^worker-loader!': '<rootDir>/tests/__mocks__/workerMock.ts'
  },

  // Transform patterns for different file types
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: 'tsconfig.json',
      diagnostics: {
        ignoreCodes: ['TS151001']
      }
    }],
    '^.+\\.jsx?$': ['babel-jest', {
      presets: ['@babel/preset-env', '@babel/preset-react']
    }]
  },

  // Coverage configuration
  coverageDirectory: '<rootDir>/coverage',
  collectCoverage: true,
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/vite-env.d.ts',
    '!src/main.tsx',
    '!src/**/*.stories.{ts,tsx}',
    '!src/**/__mocks__/**',
    '!src/**/types/**'
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  },

  // Test patterns
  testMatch: [
    '<rootDir>/tests/**/*.test.{ts,tsx}',
    '<rootDir>/tests/**/*.spec.{ts,tsx}',
    '<rootDir>/tests/**/*.perf.{ts,tsx}' // Performance-specific tests
  ],

  // Test environment configuration
  testEnvironmentOptions: {
    url: 'http://localhost',
    customExportConditions: ['node', 'node-addons'],
    resources: 'usable'
  },

  // Performance timeouts
  testTimeout: 10000,
  slowTestThreshold: 5000,

  // Snapshot configuration
  snapshotSerializers: [
    'jest-serializer-html'
  ],

  // Module file extensions
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],

  // Global configuration
  globals: {
    'ts-jest': {
      tsconfig: 'tsconfig.json',
      isolatedModules: true
    }
  },

  // Reporter configuration
  reporters: [
    'default',
    ['jest-junit', {
      outputDirectory: 'reports/junit',
      outputName: 'jest-junit.xml',
      classNameTemplate: '{classname}',
      titleTemplate: '{title}',
      ancestorSeparator: ' › ',
      usePathForSuiteName: true
    }]
  ],

  // Verbose output for debugging
  verbose: true,

  // Clear mocks between tests
  clearMocks: true,
  resetMocks: false,
  restoreMocks: false,

  // Detect memory leaks
  detectLeaks: true,
  detectOpenHandles: true
};

export default config;