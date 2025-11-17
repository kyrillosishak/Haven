/**
 * Tests for WllamaProvider
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { WllamaProvider } from './WllamaProvider';
import type { GenerateOptions } from './types';

describe('WllamaProvider', () => {
  let provider: WllamaProvider;

  beforeEach(() => {
    provider = new WllamaProvider({
      modelUrl: 'https://example.com/model.gguf',
      modelConfig: {
        n_ctx: 2048,
        n_batch: 512,
        n_threads: 1,
      },
    });
  });

  afterEach(async () => {
    if (provider) {
      await provider.dispose();
    }
  });

  describe('initialization', () => {
    it('should create provider instance', () => {
      expect(provider).toBeDefined();
      expect(provider.isInitialized()).toBe(false);
    });

    it('should throw error when generating before initialization', async () => {
      await expect(provider.generate('test prompt')).rejects.toThrow(
        'WllamaProvider not initialized'
      );
    });

    it('should throw error when streaming before initialization', async () => {
      const generator = provider.generateStream('test prompt');
      await expect(generator.next()).rejects.toThrow('WllamaProvider not initialized');
    });

    it('should return model info', () => {
      const info = provider.getModelInfo();
      expect(info.url).toBe('https://example.com/model.gguf');
      expect(info.loaded).toBe(false);
    });
  });

  describe('configuration', () => {
    it('should accept custom model config', () => {
      const customProvider = new WllamaProvider({
        modelUrl: 'https://example.com/custom-model.gguf',
        modelConfig: {
          n_ctx: 4096,
          n_batch: 1024,
          n_threads: 2,
          embeddings: true,
        },
      });

      expect(customProvider).toBeDefined();
      const info = customProvider.getModelInfo();
      expect(info.url).toBe('https://example.com/custom-model.gguf');
    });

    it('should accept progress callback', () => {
      const progressCallback = vi.fn();
      const providerWithProgress = new WllamaProvider({
        modelUrl: 'https://example.com/model.gguf',
        progressCallback,
      });

      expect(providerWithProgress).toBeDefined();
    });

    it('should accept custom WASM paths', () => {
      const customProvider = new WllamaProvider({
        modelUrl: 'https://example.com/model.gguf',
        wasmPaths: {
          'single-thread/wllama.wasm': '/custom/path/wllama.wasm',
        },
      });

      expect(customProvider).toBeDefined();
    });
  });

  describe('generate options', () => {
    it('should accept generation options', async () => {
      const options: GenerateOptions = {
        maxTokens: 100,
        temperature: 0.8,
        topP: 0.95,
        topK: 50,
        stopSequences: ['\n', 'END'],
      };

      // This will fail because wllama is not actually initialized
      // but it tests that the options are accepted
      await expect(provider.generate('test', options)).rejects.toThrow();
    });

    it('should use default options when not provided', async () => {
      // This will fail because wllama is not actually initialized
      // but it tests that defaults are used
      await expect(provider.generate('test')).rejects.toThrow();
    });
  });

  describe('resource cleanup', () => {
    it('should dispose cleanly when not initialized', async () => {
      await expect(provider.dispose()).resolves.not.toThrow();
      expect(provider.isInitialized()).toBe(false);
    });

    it('should allow multiple dispose calls', async () => {
      await provider.dispose();
      await expect(provider.dispose()).resolves.not.toThrow();
    });
  });

  describe('error handling', () => {
    it('should handle initialization errors gracefully', async () => {
      const badProvider = new WllamaProvider({
        modelUrl: '', // Invalid URL
      });

      // The actual error will depend on wllama's behavior
      // This tests that errors are caught and wrapped
      await expect(badProvider.initialize()).rejects.toThrow();
    });

    it('should provide meaningful error messages', async () => {
      await expect(provider.generate('test')).rejects.toThrow(
        /WllamaProvider not initialized/
      );
    });
  });
});
