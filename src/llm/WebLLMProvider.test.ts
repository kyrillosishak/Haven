/**
 * Tests for WebLLMProvider
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { WebLLMProvider } from './WebLLMProvider';
import type { GenerateOptions } from './types';
import { createMockMLCEngine } from '../test/mocks/webllm.js';

// Mock @mlc-ai/web-llm module
vi.mock('@mlc-ai/web-llm', () => ({
  CreateMLCEngine: vi.fn().mockImplementation(async (model: string, config?: any) => {
    // Simulate initialization progress callbacks
    if (config?.initProgressCallback) {
      config.initProgressCallback({ 
        progress: 0.3, 
        text: 'Loading model...',
      });
      config.initProgressCallback({ 
        progress: 0.7, 
        text: 'Initializing...',
      });
      config.initProgressCallback({ 
        progress: 1.0, 
        text: 'Model loaded',
      });
    }
    
    return createMockMLCEngine();
  }),
}));

// Mock navigator.gpu for WebGPU availability tests
const mockGPU = {
  requestAdapter: vi.fn().mockResolvedValue({}),
};

Object.defineProperty(global.navigator, 'gpu', {
  writable: true,
  configurable: true,
  value: mockGPU,
});

// Extend Navigator type for tests
declare global {
  interface Navigator {
    gpu?: {
      requestAdapter: () => Promise<any>;
    };
  }
}

describe('WebLLMProvider', () => {
  let provider: WebLLMProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    
    // Reset GPU mock to available state
    mockGPU.requestAdapter.mockResolvedValue({});
    
    provider = new WebLLMProvider({
      model: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
      engineConfig: {
        logLevel: 'ERROR',
      },
      chatConfig: {
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 512,
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

    it('should initialize successfully with mocked engine', async () => {
      await provider.initialize();
      
      expect(provider.isInitialized()).toBe(true);
      
      const { CreateMLCEngine } = await import('@mlc-ai/web-llm');
      expect(CreateMLCEngine).toHaveBeenCalledWith(
        'Llama-3.2-1B-Instruct-q4f32_1-MLC',
        expect.objectContaining({
          logLevel: 'ERROR',
        })
      );
    });

    it('should call progress callback during initialization', async () => {
      const progressCallback = vi.fn();
      const providerWithProgress = new WebLLMProvider({
        model: 'test-model',
        engineConfig: {
          initProgressCallback: progressCallback,
        },
      });

      await providerWithProgress.initialize();

      expect(progressCallback).toHaveBeenCalledWith(
        expect.objectContaining({
          progress: expect.any(Number),
          text: expect.any(String),
        })
      );
      expect(progressCallback).toHaveBeenCalledTimes(3);
      
      await providerWithProgress.dispose();
    });

    it('should throw error when generating before initialization', async () => {
      await expect(provider.generate('test prompt')).rejects.toThrow(
        'WebLLMProvider not initialized'
      );
    });

    it('should throw error when streaming before initialization', async () => {
      const generator = provider.generateStream('test prompt');
      await expect(generator.next()).rejects.toThrow('WebLLMProvider not initialized');
    });

    it('should return model info', () => {
      const info = provider.getModelInfo();
      expect(info.model).toBe('Llama-3.2-1B-Instruct-q4f32_1-MLC');
      expect(info.initialized).toBe(false);
      expect(info.webGPUAvailable).toBe(false);
    });

    it('should not reinitialize if already initialized', async () => {
      await provider.initialize();
      const { CreateMLCEngine } = await import('@mlc-ai/web-llm');
      const callCount = vi.mocked(CreateMLCEngine).mock.calls.length;

      await provider.initialize();
      
      expect(vi.mocked(CreateMLCEngine).mock.calls.length).toBe(callCount);
    });
  });

  describe('WebGPU availability', () => {
    it('should check WebGPU availability when available', async () => {
      mockGPU.requestAdapter.mockResolvedValue({});
      
      const available = await WebLLMProvider.isWebGPUAvailable();
      expect(available).toBe(true);
      expect(mockGPU.requestAdapter).toHaveBeenCalled();
    });

    it('should detect when WebGPU is not available', async () => {
      mockGPU.requestAdapter.mockResolvedValue(null);
      
      const available = await WebLLMProvider.isWebGPUAvailable();
      expect(available).toBe(false);
    });

    it('should handle WebGPU unavailability during initialization', async () => {
      mockGPU.requestAdapter.mockResolvedValue(null);
      
      const newProvider = new WebLLMProvider({
        model: 'test-model',
      });

      await expect(newProvider.initialize()).rejects.toThrow(/WebGPU/);
      await newProvider.dispose();
    });

    it('should suggest WllamaProvider fallback when WebGPU unavailable', async () => {
      mockGPU.requestAdapter.mockResolvedValue(null);
      
      const newProvider = new WebLLMProvider({
        model: 'test-model',
      });

      try {
        await newProvider.initialize();
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
        if (error instanceof Error) {
          expect(error.message).toMatch(/WllamaProvider/);
        }
      }
      
      await newProvider.dispose();
    });

    it('should handle navigator.gpu not being defined', async () => {
      const originalGPU = global.navigator.gpu;
      // @ts-ignore - Testing undefined case
      delete global.navigator.gpu;
      
      const available = await WebLLMProvider.isWebGPUAvailable();
      expect(available).toBe(false);
      
      // Restore
      Object.defineProperty(global.navigator, 'gpu', {
        writable: true,
        configurable: true,
        value: originalGPU,
      });
    });
  });

  describe('configuration', () => {
    it('should accept custom engine config', () => {
      const customProvider = new WebLLMProvider({
        model: 'custom-model',
        engineConfig: {
          logLevel: 'DEBUG',
          initProgressCallback: vi.fn(),
        },
      });

      expect(customProvider).toBeDefined();
      const info = customProvider.getModelInfo();
      expect(info.model).toBe('custom-model');
    });

    it('should accept custom chat config', () => {
      const customProvider = new WebLLMProvider({
        model: 'test-model',
        chatConfig: {
          temperature: 0.5,
          top_p: 0.8,
          max_tokens: 1024,
          frequency_penalty: 0.5,
          presence_penalty: 0.5,
        },
      });

      expect(customProvider).toBeDefined();
    });

    it('should accept progress callback', () => {
      const progressCallback = vi.fn();
      const providerWithProgress = new WebLLMProvider({
        model: 'test-model',
        engineConfig: {
          initProgressCallback: progressCallback,
        },
      });

      expect(providerWithProgress).toBeDefined();
    });
  });

  describe('text generation', () => {
    beforeEach(async () => {
      await provider.initialize();
    });

    it('should generate text successfully', async () => {
      const result = await provider.generate('Hello, world!');
      
      expect(result).toBeDefined();
      expect(typeof result).toBe('string');
      expect(result.length).toBeGreaterThan(0);
    });

    it('should generate deterministic responses for same prompt', async () => {
      const prompt = 'What is AI?';
      const result1 = await provider.generate(prompt);
      const result2 = await provider.generate(prompt);
      
      expect(result1).toBe(result2);
    });

    it('should accept generation options', async () => {
      const options: GenerateOptions = {
        maxTokens: 100,
        temperature: 0.8,
        topP: 0.95,
        stopSequences: ['\n', 'END'],
      };

      const result = await provider.generate('test prompt', options);
      
      expect(result).toBeDefined();
      expect(typeof result).toBe('string');
    });

    it('should use default options when not provided', async () => {
      const result = await provider.generate('test prompt');
      
      expect(result).toBeDefined();
      expect(typeof result).toBe('string');
    });

    it('should use config defaults when options not provided', async () => {
      const providerWithDefaults = new WebLLMProvider({
        model: 'test-model',
        chatConfig: {
          temperature: 0.5,
          top_p: 0.8,
          max_tokens: 256,
        },
      });

      await providerWithDefaults.initialize();
      const result = await providerWithDefaults.generate('test prompt');
      
      expect(result).toBeDefined();
      expect(typeof result).toBe('string');
      
      await providerWithDefaults.dispose();
    });

    it('should handle different prompt types', async () => {
      const prompts = [
        'Hello',
        'What is the meaning of life?',
        'Explain quantum physics',
      ];

      for (const prompt of prompts) {
        const result = await provider.generate(prompt);
        expect(result).toBeDefined();
        expect(typeof result).toBe('string');
      }
    });
  });

  describe('streaming generation', () => {
    beforeEach(async () => {
      await provider.initialize();
    });

    it('should generate streaming text', async () => {
      const chunks: string[] = [];
      const generator = provider.generateStream('Hello, world!');

      for await (const chunk of generator) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(0);
      expect(chunks.every(chunk => typeof chunk === 'string')).toBe(true);
      
      const fullText = chunks.join('');
      expect(fullText.length).toBeGreaterThan(0);
    });

    it('should stream multiple chunks', async () => {
      const chunks: string[] = [];
      const generator = provider.generateStream('Explain artificial intelligence');

      for await (const chunk of generator) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(1);
    });

    it('should accept options for streaming', async () => {
      const options: GenerateOptions = {
        maxTokens: 50,
        temperature: 0.5,
      };

      const chunks: string[] = [];
      const generator = provider.generateStream('test prompt', options);

      for await (const chunk of generator) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(0);
    });

    it('should produce same content as non-streaming for same prompt', async () => {
      const prompt = 'What is machine learning?';
      
      const nonStreamingResult = await provider.generate(prompt);
      
      const chunks: string[] = [];
      const generator = provider.generateStream(prompt);
      for await (const chunk of generator) {
        chunks.push(chunk);
      }
      const streamingResult = chunks.join('');

      expect(streamingResult.trim()).toBe(nonStreamingResult.trim());
    });
  });

  describe('resource cleanup', () => {
    it('should dispose cleanly when not initialized', async () => {
      await expect(provider.dispose()).resolves.not.toThrow();
      expect(provider.isInitialized()).toBe(false);
    });

    it('should dispose cleanly after initialization', async () => {
      await provider.initialize();
      expect(provider.isInitialized()).toBe(true);
      
      await provider.dispose();
      expect(provider.isInitialized()).toBe(false);
    });

    it('should allow multiple dispose calls', async () => {
      await provider.dispose();
      await expect(provider.dispose()).resolves.not.toThrow();
    });

    it('should not allow generation after disposal', async () => {
      await provider.initialize();
      await provider.dispose();
      
      await expect(provider.generate('test')).rejects.toThrow(
        'WebLLMProvider not initialized'
      );
    });
  });

  describe('error handling', () => {
    it('should provide meaningful error messages when not initialized', async () => {
      await expect(provider.generate('test')).rejects.toThrow(
        /WebLLMProvider not initialized/
      );
    });

    it('should handle streaming errors when not initialized', async () => {
      const generator = provider.generateStream('test');
      await expect(generator.next()).rejects.toThrow(
        'WebLLMProvider not initialized'
      );
    });

    it('should handle WebGPU initialization errors', async () => {
      mockGPU.requestAdapter.mockResolvedValue(null);
      
      const newProvider = new WebLLMProvider({
        model: 'test-model',
      });

      await expect(newProvider.initialize()).rejects.toThrow(/WebGPU/);
      await newProvider.dispose();
    });

    it('should handle engine creation errors', async () => {
      const { CreateMLCEngine } = await import('@mlc-ai/web-llm');
      vi.mocked(CreateMLCEngine).mockRejectedValueOnce(new Error('Engine creation failed'));
      
      const newProvider = new WebLLMProvider({
        model: 'test-model',
      });

      await expect(newProvider.initialize()).rejects.toThrow(/Failed to initialize/);
      await newProvider.dispose();
    });
  });

  describe('additional features', () => {
    it('should return null for runtime stats before initialization', async () => {
      const stats = await provider.getRuntimeStats();
      expect(stats).toBeNull();
    });

    it('should return runtime stats after initialization', async () => {
      await provider.initialize();
      const stats = await provider.getRuntimeStats();
      
      expect(stats).toBeDefined();
      expect(typeof stats).toBe('string');
      expect(stats).toContain('Mock Runtime Stats');
    });

    it('should throw when resetting chat before initialization', async () => {
      await expect(provider.resetChat()).rejects.toThrow(
        'WebLLMProvider not initialized'
      );
    });

    it('should reset chat successfully after initialization', async () => {
      await provider.initialize();
      
      await provider.generate('First message');
      await expect(provider.resetChat()).resolves.not.toThrow();
    });

    it('should clear chat history on reset', async () => {
      await provider.initialize();
      
      await provider.generate('Message 1');
      await provider.generate('Message 2');
      await provider.resetChat();
      
      // After reset, should be able to generate again
      const result = await provider.generate('New conversation');
      expect(result).toBeDefined();
    });
  });

  describe('OpenAI-compatible API', () => {
    beforeEach(async () => {
      await provider.initialize();
    });

    it('should format prompts as OpenAI messages', async () => {
      const result = await provider.generate('Hello, world!');
      
      expect(result).toBeDefined();
      expect(typeof result).toBe('string');
    });

    it('should support streaming with OpenAI format', async () => {
      const chunks: string[] = [];
      const generator = provider.generateStream('Hello, world!');

      for await (const chunk of generator) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(0);
    });

    it('should handle multi-turn conversations', async () => {
      const response1 = await provider.generate('What is AI?');
      expect(response1).toBeDefined();
      
      const response2 = await provider.generate('Tell me more');
      expect(response2).toBeDefined();
      
      // Both responses should be valid
      expect(response1.length).toBeGreaterThan(0);
      expect(response2.length).toBeGreaterThan(0);
    });
  });
});
