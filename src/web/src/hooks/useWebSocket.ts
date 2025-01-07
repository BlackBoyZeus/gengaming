import { useState, useEffect, useCallback, useRef } from 'react'; // v18.0.0
import {
  WS_CONFIG,
  WS_MESSAGE_TYPES,
  validateMessage,
  createWebSocketURL
} from '../config/websocket';

interface WSMessage {
  type: keyof typeof WS_MESSAGE_TYPES;
  data: unknown;
  timestamp: number;
}

interface WebSocketOptions {
  onMessage?: (message: WSMessage) => void;
  onBinaryMessage?: (data: ArrayBuffer) => void;
  onError?: (error: Error) => void;
  onClose?: () => void;
  onOpen?: () => void;
  reconnectInterval?: number;
  maxRetries?: number;
  pingInterval?: number;
  frameBufferSize?: number;
}

interface WebSocketState {
  isConnected: boolean;
  isConnecting: boolean;
  error: Error | null;
  retryCount: number;
  latency: number;
  fps: number;
  bufferSize: number;
}

/**
 * Enhanced React hook for managing WebSocket connections with performance monitoring
 * and automatic reconnection capabilities.
 * @version 1.0.0
 */
export function useWebSocket(
  path: string,
  params: Record<string, string> = {},
  options: WebSocketOptions = {}
) {
  // Connection state management
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    retryCount: 0,
    latency: 0,
    fps: 0,
    bufferSize: 0
  });

  // Mutable refs for WebSocket instance and timers
  const wsRef = useRef<WebSocket | null>(null);
  const pingTimerRef = useRef<number>();
  const reconnectTimerRef = useRef<number>();
  const frameBufferRef = useRef<ArrayBuffer[]>([]);
  const lastPingRef = useRef<number>(0);
  const fpsCounterRef = useRef<{ frames: number; timestamp: number }>({
    frames: 0,
    timestamp: Date.now()
  });

  // Configuration with defaults
  const {
    onMessage,
    onBinaryMessage,
    onError,
    onClose,
    onOpen,
    reconnectInterval = WS_CONFIG.RECONNECT_INTERVAL,
    maxRetries = WS_CONFIG.MAX_RETRIES,
    pingInterval = WS_CONFIG.PING_INTERVAL,
    frameBufferSize = WS_CONFIG.FRAME_BUFFER_SIZE
  } = options;

  /**
   * Updates FPS counter based on received frames
   */
  const updateFPS = useCallback(() => {
    const now = Date.now();
    const counter = fpsCounterRef.current;
    counter.frames++;

    if (now - counter.timestamp >= 1000) {
      setState(prev => ({
        ...prev,
        fps: counter.frames
      }));
      counter.frames = 0;
      counter.timestamp = now;
    }
  }, []);

  /**
   * Handles incoming WebSocket messages with type validation
   */
  const handleMessage = useCallback((event: MessageEvent) => {
    if (event.data instanceof ArrayBuffer) {
      updateFPS();
      frameBufferRef.current.push(event.data);
      if (frameBufferRef.current.length > frameBufferSize) {
        frameBufferRef.current.shift();
      }
      onBinaryMessage?.(event.data);
      setState(prev => ({ ...prev, bufferSize: frameBufferRef.current.length }));
      return;
    }

    try {
      const message = JSON.parse(event.data) as WSMessage;
      const validation = validateMessage(message, {
        allowedTypes: Object.values(WS_MESSAGE_TYPES),
        requiredFields: ['type', 'data', 'timestamp']
      });

      if (!validation.valid) {
        throw new Error(`Invalid message format: ${validation.errors.join(', ')}`);
      }

      if (message.type === WS_MESSAGE_TYPES.PONG) {
        const latency = Date.now() - lastPingRef.current;
        setState(prev => ({ ...prev, latency }));
      } else {
        onMessage?.(message);
      }
    } catch (error) {
      console.error('WebSocket message parsing error:', error);
    }
  }, [onMessage, onBinaryMessage, frameBufferSize, updateFPS]);

  /**
   * Establishes WebSocket connection with security validation
   */
  const connect = useCallback(() => {
    try {
      const url = createWebSocketURL(path, params, {
        secure: window.location.protocol === 'https:'
      });

      setState(prev => ({ ...prev, isConnecting: true }));
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        setState(prev => ({
          ...prev,
          isConnected: true,
          isConnecting: false,
          error: null,
          retryCount: 0
        }));
        onOpen?.();
      };

      ws.onmessage = handleMessage;

      ws.onerror = (event) => {
        const error = new Error('WebSocket error occurred');
        setState(prev => ({ ...prev, error }));
        onError?.(error);
      };

      ws.onclose = (event) => {
        setState(prev => ({
          ...prev,
          isConnected: false,
          isConnecting: false
        }));
        onClose?.();

        if (state.retryCount < maxRetries) {
          reconnectTimerRef.current = window.setTimeout(() => {
            setState(prev => ({ ...prev, retryCount: prev.retryCount + 1 }));
            connect();
          }, reconnectInterval * Math.pow(2, state.retryCount));
        }
      };
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error as Error,
        isConnecting: false
      }));
    }
  }, [path, params, maxRetries, reconnectInterval, state.retryCount, handleMessage, onOpen, onError, onClose]);

  /**
   * Sends message with rate limiting and validation
   */
  const sendMessage = useCallback((type: keyof typeof WS_MESSAGE_TYPES, data: unknown) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return false;
    }

    const message = {
      type,
      data,
      timestamp: Date.now()
    };

    const validation = validateMessage(message);
    if (!validation.valid) {
      console.error('Invalid message:', validation.errors);
      return false;
    }

    wsRef.current.send(JSON.stringify(message));
    return true;
  }, []);

  /**
   * Sends binary message with size validation
   */
  const sendBinaryMessage = useCallback((data: ArrayBuffer) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return false;
    }

    if (data.byteLength > WS_CONFIG.MAX_MESSAGE_SIZE) {
      console.error('Binary message exceeds size limit');
      return false;
    }

    wsRef.current.send(data);
    return true;
  }, []);

  /**
   * Initiates connection health check
   */
  const startPingInterval = useCallback(() => {
    pingTimerRef.current = window.setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        lastPingRef.current = Date.now();
        sendMessage(WS_MESSAGE_TYPES.PING, null);
      }
    }, pingInterval);
  }, [pingInterval, sendMessage]);

  // Setup effect
  useEffect(() => {
    connect();
    startPingInterval();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (pingTimerRef.current) clearInterval(pingTimerRef.current);
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    };
  }, [connect, startPingInterval]);

  return {
    ...state,
    sendMessage,
    sendBinaryMessage,
    reconnect: connect,
    disconnect: useCallback(() => {
      wsRef.current?.close();
    }, [])
  };
}