// @testing-library/jest-dom v5.16.5
import '@testing-library/jest-dom';
// jest-environment-jsdom v29.5.0
import 'jest-environment-jsdom';
// resize-observer-polyfill v1.5.1
import ResizeObserverPolyfill from 'resize-observer-polyfill';

// Configure global mocks and polyfills
global.ResizeObserver = ResizeObserverPolyfill;

// Mock matchMedia for responsive design testing
global.matchMedia = function mockMatchMedia(query: string): MediaQueryList {
  return {
    matches: false, // Default to not matching
    media: query,
    onchange: null,
    addListener: jest.fn(), // Deprecated but included for legacy support
    removeListener: jest.fn(), // Deprecated but included for legacy support
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  };
};

// Mock WebSocket for real-time feature testing with latency simulation
class MockWebSocket {
  private url: string;
  private protocols: string[];
  private readyState: number;
  private messageQueue: any[];
  private latency: number;

  public onopen: ((event: any) => void) | null;
  public onmessage: ((event: any) => void) | null;
  public onclose: ((event: any) => void) | null;
  public onerror: ((event: any) => void) | null;

  constructor(url: string, protocols: string[] = []) {
    this.url = url;
    this.protocols = protocols;
    this.readyState = WebSocket.CONNECTING;
    this.messageQueue = [];
    this.latency = 50; // Default 50ms latency

    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
    this.onerror = null;

    // Simulate connection establishment
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) {
        this.onopen({ type: 'open' });
      }
    }, this.latency);
  }

  send(data: any): void {
    if (this.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }
    this.messageQueue.push(data);
  }

  close(code?: number, reason?: string): void {
    this.readyState = WebSocket.CLOSING;
    setTimeout(() => {
      this.readyState = WebSocket.CLOSED;
      if (this.onclose) {
        this.onclose({ 
          code: code || 1000,
          reason: reason || '',
          wasClean: true,
          type: 'close'
        });
      }
    }, this.latency);
  }

  // Test helper methods
  simulateMessage(data: any): void {
    if (this.onmessage) {
      setTimeout(() => {
        this.onmessage!({ data, type: 'message' });
      }, this.latency);
    }
  }

  simulateError(error: any): void {
    if (this.onerror) {
      this.onerror({ error, type: 'error' });
    }
  }

  setLatency(ms: number): void {
    this.latency = ms;
  }

  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
}

global.WebSocket = MockWebSocket as any;

// Mock Performance API for timing measurements
const performanceMock = {
  timeOrigin: Date.now(),
  marks: new Map<string, number>(),
  measures: new Map<string, { start: number; duration: number }>(),

  now(): number {
    return Date.now() - this.timeOrigin;
  },

  mark(name: string): void {
    this.marks.set(name, this.now());
  },

  measure(name: string, startMark?: string, endMark?: string): void {
    const start = startMark ? this.marks.get(startMark) : 0;
    const end = endMark ? this.marks.get(endMark) : this.now();
    
    if (startMark && !start) {
      throw new Error(`Start mark "${startMark}" not found`);
    }
    if (endMark && !end) {
      throw new Error(`End mark "${endMark}" not found`);
    }

    this.measures.set(name, {
      start: start || 0,
      duration: (end || this.now()) - (start || 0)
    });
  },

  getEntriesByType(type: string): PerformanceEntry[] {
    if (type === 'mark') {
      return Array.from(this.marks.entries()).map(([name, startTime]) => ({
        name,
        entryType: 'mark',
        startTime,
        duration: 0
      }));
    }
    if (type === 'measure') {
      return Array.from(this.measures.entries()).map(([name, data]) => ({
        name,
        entryType: 'measure',
        startTime: data.start,
        duration: data.duration
      }));
    }
    return [];
  },

  clearMarks(markName?: string): void {
    if (markName) {
      this.marks.delete(markName);
    } else {
      this.marks.clear();
    }
  },

  clearMeasures(measureName?: string): void {
    if (measureName) {
      this.measures.delete(measureName);
    } else {
      this.measures.clear();
    }
  }
};

global.performance = performanceMock as any;

// Mock requestAnimationFrame for 60fps timing simulation
const RAF_INTERVAL = 1000 / 60; // ~16.67ms for 60fps
let rafHandle = 0;

global.requestAnimationFrame = function(callback: FrameRequestCallback): number {
  const nextHandle = ++rafHandle;
  setTimeout(() => callback(performance.now()), RAF_INTERVAL);
  return nextHandle;
};

global.cancelAnimationFrame = function(handle: number): void {
  // Implementation not needed for most tests
};

// Configure Jest environment
beforeEach(() => {
  // Reset performance measurements before each test
  performanceMock.clearMarks();
  performanceMock.clearMeasures();
  
  // Reset RAF handle counter
  rafHandle = 0;
});

afterEach(() => {
  // Clean up any remaining timers
  jest.clearAllTimers();
  
  // Reset any mocked functions
  jest.clearAllMocks();
});