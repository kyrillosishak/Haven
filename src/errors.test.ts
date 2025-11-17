/**
 * Tests for error classes
 */

import { describe, it, expect, vi } from 'vitest';
import {
  VectorDBError,
  StorageQuotaError,
  DimensionMismatchError,
  ModelLoadError,
  IndexCorruptedError,
  InputValidator,
  ErrorHandler,
  DEFAULT_RETRY_CONFIG,
} from './errors';

describe('Error Classes', () => {
  it('should create VectorDBError with code and details', () => {
    const error = new VectorDBError('Test error', 'TEST_CODE', { foo: 'bar' });
    expect(error).toBeInstanceOf(Error);
    expect(error).toBeInstanceOf(VectorDBError);
    expect(error.message).toBe('Test error');
    expect(error.code).toBe('TEST_CODE');
    expect(error.details).toEqual({ foo: 'bar' });
  });

  it('should create StorageQuotaError', () => {
    const error = new StorageQuotaError({ limit: 100 });
    expect(error).toBeInstanceOf(VectorDBError);
    expect(error.code).toBe('STORAGE_QUOTA_EXCEEDED');
    expect(error.message).toContain('Storage quota exceeded');
  });

  it('should create DimensionMismatchError', () => {
    const error = new DimensionMismatchError(384, 512);
    expect(error).toBeInstanceOf(VectorDBError);
    expect(error.code).toBe('DIMENSION_MISMATCH');
    expect(error.message).toContain('384');
    expect(error.message).toContain('512');
  });

  it('should create ModelLoadError', () => {
    const cause = new Error('Network error');
    const error = new ModelLoadError('test-model', cause);
    expect(error).toBeInstanceOf(VectorDBError);
    expect(error.code).toBe('MODEL_LOAD_ERROR');
    expect(error.message).toContain('test-model');
  });

  it('should create IndexCorruptedError', () => {
    const error = new IndexCorruptedError({ reason: 'invalid format' });
    expect(error).toBeInstanceOf(VectorDBError);
    expect(error.code).toBe('INDEX_CORRUPTED');
  });
});

describe('InputValidator', () => {
  describe('validateVector', () => {
    it('should validate correct vector', () => {
      const vector = new Float32Array([1.0, 2.0, 3.0]);
      expect(() => InputValidator.validateVector(vector, 3)).not.toThrow();
    });

    it('should throw on dimension mismatch', () => {
      const vector = new Float32Array([1.0, 2.0, 3.0]);
      expect(() => InputValidator.validateVector(vector, 5)).toThrow(DimensionMismatchError);
    });

    it('should throw on NaN values', () => {
      const vector = new Float32Array([1.0, NaN, 3.0]);
      expect(() => InputValidator.validateVector(vector, 3)).toThrow(VectorDBError);
    });

    it('should throw on Infinity values', () => {
      const vector = new Float32Array([1.0, Infinity, 3.0]);
      expect(() => InputValidator.validateVector(vector, 3)).toThrow(VectorDBError);
    });
  });

  describe('validateAndSanitizeMetadata', () => {
    it('should return empty object for null/undefined', () => {
      expect(InputValidator.validateAndSanitizeMetadata(null)).toEqual({});
      expect(InputValidator.validateAndSanitizeMetadata(undefined)).toEqual({});
    });

    it('should sanitize string values to prevent XSS', () => {
      const metadata = {
        title: '<script>alert("xss")</script>',
        content: 'Hello <b>world</b>',
      };
      const sanitized = InputValidator.validateAndSanitizeMetadata(metadata);
      expect(sanitized.title).not.toContain('<script>');
      expect(sanitized.title).toContain('&lt;script&gt;');
      expect(sanitized.content).toContain('&lt;b&gt;');
    });

    it('should preserve numbers and booleans', () => {
      const metadata = {
        count: 42,
        active: true,
        score: 3.14,
      };
      const sanitized = InputValidator.validateAndSanitizeMetadata(metadata);
      expect(sanitized.count).toBe(42);
      expect(sanitized.active).toBe(true);
      expect(sanitized.score).toBe(3.14);
    });

    it('should sanitize nested objects', () => {
      const metadata = {
        user: {
          name: '<script>evil</script>',
          age: 25,
        },
      };
      const sanitized = InputValidator.validateAndSanitizeMetadata(metadata);
      expect(sanitized.user.name).toContain('&lt;script&gt;');
      expect(sanitized.user.age).toBe(25);
    });

    it('should sanitize arrays', () => {
      const metadata = {
        tags: ['<script>tag1</script>', 'tag2', 'tag3'],
      };
      const sanitized = InputValidator.validateAndSanitizeMetadata(metadata);
      expect(sanitized.tags[0]).toContain('&lt;script&gt;');
      expect(sanitized.tags[1]).toBe('tag2');
    });

    it('should throw on invalid metadata type', () => {
      expect(() => InputValidator.validateAndSanitizeMetadata('string')).toThrow(VectorDBError);
      expect(() => InputValidator.validateAndSanitizeMetadata([])).toThrow(VectorDBError);
    });

    it('should throw on invalid metadata value types', () => {
      const metadata = {
        func: () => {},
      };
      expect(() => InputValidator.validateAndSanitizeMetadata(metadata)).toThrow(VectorDBError);
    });
  });

  describe('sanitizeString', () => {
    it('should escape HTML special characters', () => {
      expect(InputValidator.sanitizeString('<div>')).toBe('&lt;div&gt;');
      expect(InputValidator.sanitizeString('"quote"')).toBe('&quot;quote&quot;');
      expect(InputValidator.sanitizeString("'single'")).toBe('&#x27;single&#x27;');
      expect(InputValidator.sanitizeString('a/b')).toBe('a&#x2F;b');
    });

    it('should handle empty strings', () => {
      expect(InputValidator.sanitizeString('')).toBe('');
    });
  });

  describe('validateSearchQuery', () => {
    it('should validate correct k value', () => {
      expect(() => InputValidator.validateSearchQuery(5)).not.toThrow();
      expect(() => InputValidator.validateSearchQuery(100)).not.toThrow();
    });

    it('should throw on non-positive k', () => {
      expect(() => InputValidator.validateSearchQuery(0)).toThrow(VectorDBError);
      expect(() => InputValidator.validateSearchQuery(-5)).toThrow(VectorDBError);
    });

    it('should throw on non-integer k', () => {
      expect(() => InputValidator.validateSearchQuery(5.5)).toThrow(VectorDBError);
    });

    it('should throw on k too large', () => {
      expect(() => InputValidator.validateSearchQuery(20000)).toThrow(VectorDBError);
    });
  });
});

describe('ErrorHandler', () => {
  describe('handleError', () => {
    it('should handle StorageQuotaError', async () => {
      const logger = vi.fn();
      const handler = new ErrorHandler(logger);
      const error = new StorageQuotaError();

      await handler.handleError(error, 'test-context');

      expect(logger).toHaveBeenCalledWith(
        expect.stringContaining('Storage quota exceeded'),
        error,
        expect.objectContaining({ context: 'test-context' })
      );
    });

    it('should handle IndexCorruptedError', async () => {
      const logger = vi.fn();
      const handler = new ErrorHandler(logger);
      const error = new IndexCorruptedError();

      await handler.handleError(error, 'test-context');

      expect(logger).toHaveBeenCalledWith(
        expect.stringContaining('Index corrupted'),
        error,
        expect.objectContaining({ context: 'test-context' })
      );
    });

    it('should handle ModelLoadError', async () => {
      const logger = vi.fn();
      const handler = new ErrorHandler(logger);
      const error = new ModelLoadError('test-model', new Error('network'));

      await handler.handleError(error, 'test-context');

      expect(logger).toHaveBeenCalledWith(
        expect.stringContaining('Failed to load model'),
        error,
        expect.objectContaining({ context: 'test-context' })
      );
    });
  });

  describe('withRetry', () => {
    it('should succeed on first attempt', async () => {
      const handler = new ErrorHandler();
      const operation = vi.fn().mockResolvedValue('success');

      const result = await handler.withRetry(operation);

      expect(result).toBe('success');
      expect(operation).toHaveBeenCalledTimes(1);
    });

    it('should retry on transient errors', async () => {
      const handler = new ErrorHandler();
      const operation = vi.fn()
        .mockRejectedValueOnce(new Error('network error'))
        .mockResolvedValue('success');

      const result = await handler.withRetry(
        operation,
        { maxAttempts: 3, initialDelayMs: 10 },
        (error) => error.message.includes('network')
      );

      expect(result).toBe('success');
      expect(operation).toHaveBeenCalledTimes(2);
    });

    it('should not retry on non-transient errors', async () => {
      const handler = new ErrorHandler();
      const error = new DimensionMismatchError(384, 512);
      const operation = vi.fn().mockRejectedValue(error);

      await expect(
        handler.withRetry(operation, { maxAttempts: 3 })
      ).rejects.toThrow(DimensionMismatchError);

      expect(operation).toHaveBeenCalledTimes(1);
    });

    it('should throw after max attempts', async () => {
      const handler = new ErrorHandler();
      const error = new ModelLoadError('test', new Error('network'));
      const operation = vi.fn().mockRejectedValue(error);

      await expect(
        handler.withRetry(operation, { maxAttempts: 2, initialDelayMs: 10 })
      ).rejects.toThrow(ModelLoadError);

      expect(operation).toHaveBeenCalledTimes(2);
    });
  });
});
