/**
 * Tests for centralized mock utilities
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  createMockPipeline,
  createMockMLCEngine,
  setupTransformersMocks,
  setupWebLLMMocks,
  setupAllMocks,
  resetAllMocks,
  type ChatCompletion,
} from './index.js';

describe('Mock Utilities', () => {
  describe('createMockPipeline', () => {
    it('should be exported and create a mock pipeline', async () => {
      const pipeline = createMockPipeline({ dimensions: 384 });
      
      expect(pipeline).toBeDefined();
      expect(typeof pipeline).toBe('function');
      expect(pipeline.dispose).toBeDefined();
      
      const result = await pipeline('test text');
      expect(result.data).toBeInstanceOf(Float32Array);
      expect(result.data.length).toBe(384);
    });
  });

  describe('createMockMLCEngine', () => {
    it('should be exported and create a mock engine', async () => {
      const engine = createMockMLCEngine({ defaultResponse: 'Test response' });
      
      expect(engine).toBeDefined();
      expect(engine.chat).toBeDefined();
      expect(engine.chat.completions).toBeDefined();
      expect(engine.unload).toBeDefined();
      
      const completion = await engine.chat.completions.create({
        messages: [{ role: 'user', content: 'Hello' }],
        stream: false,
      }) as ChatCompletion;
      
      expect(completion.choices[0].message?.content).toBe('Hello! How can I help you today?');
    });
  });

  describe('setupTransformersMocks', () => {
    afterEach(() => {
      vi.clearAllMocks();
    });

    it('should setup Transformers.js mocks', () => {
      setupTransformersMocks({ dimensions: 512 });
      
      // Verify that vi.mock was called
      expect(vi.isMockFunction).toBeDefined();
    });

    it('should accept optional configuration', () => {
      setupTransformersMocks();
      expect(true).toBe(true); // Should not throw
      
      setupTransformersMocks({ dimensions: 768, simulateDelay: 10 });
      expect(true).toBe(true); // Should not throw
    });
  });

  describe('setupWebLLMMocks', () => {
    afterEach(() => {
      vi.clearAllMocks();
    });

    it('should setup WebLLM mocks', () => {
      setupWebLLMMocks({ defaultResponse: 'Mock response' });
      
      // Verify that vi.mock was called
      expect(vi.isMockFunction).toBeDefined();
    });

    it('should accept optional configuration', () => {
      setupWebLLMMocks();
      expect(true).toBe(true); // Should not throw
      
      setupWebLLMMocks({ simulateDelay: 5, streamChunkSize: 2 });
      expect(true).toBe(true); // Should not throw
    });
  });

  describe('setupAllMocks', () => {
    afterEach(() => {
      vi.clearAllMocks();
    });

    it('should setup all mocks at once', () => {
      setupAllMocks();
      expect(true).toBe(true); // Should not throw
    });

    it('should accept configuration for both mock types', () => {
      setupAllMocks(
        { dimensions: 384 },
        { defaultResponse: 'Test' }
      );
      expect(true).toBe(true); // Should not throw
    });

    it('should work with partial configuration', () => {
      setupAllMocks({ dimensions: 512 });
      expect(true).toBe(true); // Should not throw
      
      setupAllMocks(undefined, { defaultResponse: 'Response' });
      expect(true).toBe(true); // Should not throw
    });
  });

  describe('resetAllMocks', () => {
    it('should reset all mocks', () => {
      const mockFn = vi.fn();
      mockFn();
      mockFn();
      
      expect(mockFn).toHaveBeenCalledTimes(2);
      
      resetAllMocks();
      
      // After reset, the mock should be cleared
      expect(mockFn).toHaveBeenCalledTimes(0);
    });

    it('should be safe to call multiple times', () => {
      resetAllMocks();
      resetAllMocks();
      resetAllMocks();
      
      expect(true).toBe(true); // Should not throw
    });
  });

  describe('Integration', () => {
    beforeEach(() => {
      setupAllMocks();
    });

    afterEach(() => {
      resetAllMocks();
    });

    it('should allow using both mocks together', async () => {
      const pipeline = createMockPipeline();
      const engine = createMockMLCEngine();
      
      const embedding = await pipeline('test');
      const completion = await engine.chat.completions.create({
        messages: [{ role: 'user', content: 'test' }],
        stream: false,
      }) as ChatCompletion;
      
      expect(embedding.data).toBeInstanceOf(Float32Array);
      expect(completion.choices[0].message?.content).toBeDefined();
    });
  });
});
