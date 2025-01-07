import CryptoJS from 'crypto-js'; // ^4.1.1

// Storage configuration constants
const STORAGE_PREFIX = 'gamegen_x_';
const ENCRYPTION_KEY = process.env.STORAGE_ENCRYPTION_KEY;
const STORAGE_VERSION = '1.0';

// Custom error types for storage operations
class StorageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'StorageError';
  }
}

class EncryptionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EncryptionError';
  }
}

// Type guard for storage availability
const isStorageAvailable = (): boolean => {
  try {
    const test = '__storage_test__';
    localStorage.setItem(test, test);
    localStorage.removeItem(test);
    return true;
  } catch {
    return false;
  }
};

/**
 * Encrypts data using AES encryption with a random IV
 * @param data - String data to encrypt
 * @returns Encrypted data string with IV
 * @throws EncryptionError if encryption fails
 */
const encryptData = (data: string): string => {
  try {
    if (!ENCRYPTION_KEY) {
      throw new EncryptionError('Encryption key not configured');
    }

    const iv = CryptoJS.lib.WordArray.random(16);
    const encrypted = CryptoJS.AES.encrypt(data, ENCRYPTION_KEY, {
      iv: iv,
      mode: CryptoJS.mode.CBC,
      padding: CryptoJS.pad.Pkcs7
    });

    return `${iv.toString()}:${encrypted.toString()}`;
  } catch (error) {
    throw new EncryptionError(`Encryption failed: ${error.message}`);
  }
};

/**
 * Decrypts AES encrypted data
 * @param encryptedData - Encrypted data string with IV
 * @returns Decrypted data string
 * @throws EncryptionError if decryption fails
 */
const decryptData = (encryptedData: string): string => {
  try {
    if (!ENCRYPTION_KEY) {
      throw new EncryptionError('Encryption key not configured');
    }

    const [ivString, encryptedString] = encryptedData.split(':');
    if (!ivString || !encryptedString) {
      throw new EncryptionError('Invalid encrypted data format');
    }

    const iv = CryptoJS.enc.Hex.parse(ivString);
    const decrypted = CryptoJS.AES.decrypt(encryptedString, ENCRYPTION_KEY, {
      iv: iv,
      mode: CryptoJS.mode.CBC,
      padding: CryptoJS.pad.Pkcs7
    });

    return decrypted.toString(CryptoJS.enc.Utf8);
  } catch (error) {
    throw new EncryptionError(`Decryption failed: ${error.message}`);
  }
};

/**
 * Stores data in browser local storage with optional encryption
 * @param key - Storage key
 * @param value - Data to store
 * @param encrypt - Whether to encrypt the data
 * @throws StorageError if storage operation fails
 */
export const setItem = async (
  key: string,
  value: unknown,
  encrypt = false
): Promise<void> => {
  try {
    if (!isStorageAvailable()) {
      throw new StorageError('Local storage is not available');
    }

    if (!key) {
      throw new StorageError('Storage key is required');
    }

    const storageKey = `${STORAGE_PREFIX}${key}`;
    const storageValue = {
      version: STORAGE_VERSION,
      data: value,
      timestamp: Date.now()
    };

    let serializedValue = JSON.stringify(storageValue);
    if (encrypt) {
      serializedValue = encryptData(serializedValue);
    }

    localStorage.setItem(storageKey, serializedValue);
    window.dispatchEvent(new Event('storage'));
  } catch (error) {
    if (error.name === 'QuotaExceededError') {
      throw new StorageError('Storage quota exceeded');
    }
    throw new StorageError(`Storage operation failed: ${error.message}`);
  }
};

/**
 * Retrieves and optionally decrypts data from browser local storage
 * @param key - Storage key
 * @param encrypted - Whether the data is encrypted
 * @returns Retrieved and parsed data
 * @throws StorageError if retrieval operation fails
 */
export const getItem = async (
  key: string,
  encrypted = false
): Promise<unknown> => {
  try {
    if (!isStorageAvailable()) {
      throw new StorageError('Local storage is not available');
    }

    const storageKey = `${STORAGE_PREFIX}${key}`;
    let storedValue = localStorage.getItem(storageKey);

    if (!storedValue) {
      return null;
    }

    if (encrypted) {
      storedValue = decryptData(storedValue);
    }

    const parsedValue = JSON.parse(storedValue);
    if (parsedValue.version !== STORAGE_VERSION) {
      throw new StorageError('Storage version mismatch');
    }

    return parsedValue.data;
  } catch (error) {
    throw new StorageError(`Retrieval operation failed: ${error.message}`);
  }
};

/**
 * Removes an item from browser local storage
 * @param key - Storage key
 * @throws StorageError if removal operation fails
 */
export const removeItem = async (key: string): Promise<void> => {
  try {
    if (!isStorageAvailable()) {
      throw new StorageError('Local storage is not available');
    }

    const storageKey = `${STORAGE_PREFIX}${key}`;
    localStorage.removeItem(storageKey);
    window.dispatchEvent(new Event('storage'));
  } catch (error) {
    throw new StorageError(`Remove operation failed: ${error.message}`);
  }
};

/**
 * Clears all GameGen-X related items from local storage
 * @throws StorageError if clear operation fails
 */
export const clearStorage = async (): Promise<void> => {
  try {
    if (!isStorageAvailable()) {
      throw new StorageError('Local storage is not available');
    }

    const keys = Object.keys(localStorage);
    const gameGenKeys = keys.filter(key => key.startsWith(STORAGE_PREFIX));

    gameGenKeys.forEach(key => {
      localStorage.removeItem(key);
    });

    window.dispatchEvent(new Event('storage'));
  } catch (error) {
    throw new StorageError(`Clear operation failed: ${error.message}`);
  }
};