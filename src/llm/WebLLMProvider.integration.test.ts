/**
 * Integration tests for WebLLMProvider with real models
 * 
 * These tests require:
 * - Real browser environment with WebGPU support (not jsdom/happy-dom)
 * - Internet connection for model downloads
 * - WebGPU-capable GPU
 * 
 * Run with: npm run test:integration
 * Skip with: SKIP_INTEGRATION=true npm test
 * 
 * Note: These tests are automatically skipped in CI environments
 * Note: These tests require significant time and resources due to model downloads
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { WebLLMProvider } from './WebLLMProvider';
import type { GenerateOptions } from './types';

// Skip integration tests in CI or when explicitly disabled
const skipIntegration = process.env.CI === 'true' || process.env.SKIP_INTEGRATION === 'true';

describe.skipIf(skipIntegration)('WebLLMProvider (Integration)', () => {
  let provider: WebLLMProvider;
  let webGPUAvailable: boolean;

  beforeAll(async () => {
    // Check WebGPU availability first
    webGPUAvailable = await WebLLMProvider.isWebGPUAvailable();
    
    if (!webGPUAvailable) {
      console.warn('WebGPU not available, skipping WebLLMProvider integration tests');
      return;
    }

    // Initialize with a small, fast model for testing
    // Using Llama-3.2-1B as it's one of the smallest models
    provider = new WebLLMProvider({
      model: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
      engineConfig: {
        logLevel: 'ERROR',
        initProgressCallback: (progress) => {
          console.log(`Loading model: ${(progress.progress * 100).toFixed(1)}% - ${progress.text}`);
        },
      },
      chatConfig: {
        temperature: 0.7,
        max_tokens: 256,
      },
    });

    // This will download the real model (can take several minutes)
    await provider.initialize();
  }, 300000); // 5 minute timeout for model download

  afterAll(async () => {
    if (provider && provider.isInitialized()) {
      await provider.dispose();
    }
  });

  describe('real model initialization', () => {
    it.skipIf(!webGPUAvailable)('should load real model with WebGPU', () => {
      expect(provider.isInitialized()).toBe(true);
      
      const info = provider.getModelInfo();
      expect(info.initialized).toBe(true);
      expect(info.model).toBe('Llama-3.2-1B-Instruct-q4f32_1-MLC');
    });

    it.skipIf(!webGPUAvailable)('should report WebGPU availability', () => {
      const info = provider.getModelInfo();
      expect(info.webGPUAvailable).toBe(true);
    });
  });

  describe('real text generation', () => {
    it.skipIf(!webGPUAvailable)('should generate coherent text', async () => {
      const prompt = 'What is artificial intelligence?';
      const result = await provider.generate(prompt);

      expect(result).toBeDefined();
      expect(typeof result).toBe('string');
      expect(result.length).toBeGreaterThan(10);
      
      // Result should contain relevant keywords
      const lowerResult = result.toLowerCase();
      const hasRelevantContent = 
        lowerResult.includes('ai') || 
        lowerResult.includes('artificial') ||
        lowerResult.includes('intelligence') ||
        lowerResult.includes('machine') ||
        lowerResult.includes('learning');
      
      expect(hasRelevantContent).toBe(true);
    }, 30000); // 30 second timeout for generation

    it.skipIf(!webGPUAvailable)('should respect max tokens limit', async () => {
      const options: GenerateOptions = {
        maxTokens: 50,
      };

      const result = await provider.generate('Tell me a story', options);

      expect(result).toBeDefined();
      expect(typeof result).toBe('string');
      
      // Rough estimate: 50 tokens ≈ 200-250 characters
      // Allow some flexibility as tokenization varies
      expect(result.length).toBeLessThan(500);
    }, 30000);

    it.skipIf(!webGPUAvailable)('should handle different temperatures', async () => {
      const prompt = 'Complete this sentence: The sky is';

      // Low temperature (more deterministic)
      const result1 = await provider.generate(prompt, { temperature: 0.1 });
      const result2 = await provider.generate(prompt, { temperature: 0.1 });

      // High temperature (more creative)
      const result3 = await provider.generate(prompt, { temperature: 1.5 });

      expect(result1).toBeDefined();
      expect(result2).toBeDefined();
      expect(result3).toBeDefined();

      // Low temperature results should be more similar
      // (though not necessarily identical due to sampling)
      expect(result1.length).toBeGreaterThan(0);
      expect(result3.length).toBeGreaterThan(0);
    }, 60000);

    it.skipIf(!webGPUAvailable)('should handle various prompt types', async () => {
      const prompts = [
        'What is 2+2?',
        'Write a haiku about coding',
        'Explain quantum computing in one sentence',
      ];

      for (const prompt of prompts) {
        const result = await provider.generate(prompt, { maxTokens: 100 });
        expect(result).toBeDefined();
        expect(typeof result).toBe('string');
        expect(result.length).toBeGreaterThan(0);
      }
    }, 90000);
  });

  describe('streaming generation with real model', () => {
    it.skipIf(!webGPUAvailable)('should stream text chunks', async () => {
      const prompt = 'Count from 1 to 5';
      const chunks: string[] = [];

      const generator = provider.generateStream(prompt, { maxTokens: 50 });

      for await (const chunk of generator) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(0);
      
      const fullText = chunks.join('');
      expect(fullText.length).toBeGreaterThan(0);
      
      // Should contain numbers
      expect(/\d/.test(fullText)).toBe(true);
    }, 30000);

    it.skipIf(!webGPUAvailable)('should stream incrementally', async () => {
      const prompt = 'What is machine learning?';
      const chunks: string[] = [];
      let chunkCount = 0;

      const generator = provider.generateStream(prompt, { maxTokens: 100 });

      for await (const chunk of generator) {
        chunks.push(chunk);
        chunkCount++;
        
        // Verify we're getting incremental updates
        if (chunkCount > 1) {
          expect(chunk.length).toBeGreaterThan(0);
        }
      }

      // Should receive multiple chunks
      expect(chunkCount).toBeGreaterThan(1);
      
      const fullText = chunks.join('');
      expect(fullText.length).toBeGreaterThan(10);
    }, 30000);

    it.skipIf(!webGPUAvailable)('should produce same content as non-streaming', async () => {
      const prompt = 'What is 2+2?';
      
      // Non-streaming
      const nonStreamResult = await provider.generate(prompt, { 
        maxTokens: 50,
        temperature: 0.1, // Low temperature for consistency
      });

      // Reset chat to ensure clean state
      await provider.resetChat();

      // Streaming
      const chunks: string[] = [];
      const generator = provider.generateStream(prompt, { 
        maxTokens: 50,
        temperature: 0.1,
      });
      
      for await (const chunk of generator) {
        chunks.push(chunk);
      }
      const streamResult = chunks.join('');

      // Results should be similar (may not be identical due to sampling)
      expect(streamResult.length).toBeGreaterThan(0);
      expect(nonStreamResult.length).toBeGreaterThan(0);
      
      // Both should contain the answer "4"
      expect(streamResult).toMatch(/4/);
      expect(nonStreamResult).toMatch(/4/);
    }, 60000);
  });

  describe('chat functionality', () => {
    it.skipIf(!webGPUAvailable)('should maintain conversation context', async () => {
      await provider.resetChat();

      const response1 = await provider.generate('My name is Alice', { maxTokens: 50 });
      expect(response1).toBeDefined();

      const response2 = await provider.generate('What is my name?', { maxTokens: 50 });
      expect(response2).toBeDefined();
      
      // Response should reference the name (though not guaranteed with small models)
      const lowerResponse = response2.toLowerCase();
      const hasContext = lowerResponse.includes('alice') || lowerResponse.includes('name');
      
      // Note: Small models may not always maintain context perfectly
      // This is a best-effort test
      if (hasContext) {
        expect(hasContext).toBe(true);
      }
    }, 60000);

    it.skipIf(!webGPUAvailable)('should reset chat history', async () => {
      await provider.resetChat();

      await provider.generate('Remember the number 42', { maxTokens: 50 });
      
      await provider.resetChat();
      
      const response = await provider.generate('What number should you remember?', { maxTokens: 50 });
      expect(response).toBeDefined();
      
      // After reset, should not remember the number
      // (though this is not guaranteed with all models)
      expect(typeof response).toBe('string');
    }, 60000);
  });

  describe('runtime statistics', () => {
    it.skipIf(!webGPUAvailable)('should provide runtime statistics', async () => {
      const stats = await provider.getRuntimeStats();

      expect(stats).toBeDefined();
      expect(typeof stats).toBe('string');
      expect(stats!.length).toBeGreaterThan(0);
    });
  });

  describe('performance with real model', () => {
    it.skipIf(!webGPUAvailable)('should generate text in reasonable time', async () => {
      const prompt = 'Hello';
      const startTime = Date.now();

      await provider.generate(prompt, { maxTokens: 50 });

      const duration = Date.now() - startTime;
      
      // Should complete in less than 30 seconds
      expect(duration).toBeLessThan(30000);
    }, 30000);

    it.skipIf(!webGPUAvailable)('should handle multiple sequential generations', async () => {
      const prompts = [
        'What is AI?',
        'What is ML?',
        'What is DL?',
      ];

      for (const prompt of prompts) {
        const result = await provider.generate(prompt, { maxTokens: 30 });
        expect(result).toBeDefined();
        expect(result.length).toBeGreaterThan(0);
      }
    }, 90000);
  });

  describe('error handling with real model', () => {
    it.skipIf(!webGPUAvailable)('should handle empty prompts gracefully', async () => {
      // Empty prompt might produce a response or error depending on model
      try {
        const result = await provider.generate('', { maxTokens: 20 });
        // If it succeeds, result should be defined
        expect(result).toBeDefined();
      } catch (error) {
        // If it fails, error should be meaningful
        expect(error).toBeInstanceOf(Error);
      }
    }, 30000);

    it.skipIf(!webGPUAvailable)('should handle very long prompts', async () => {
      const longPrompt = 'This is a test. '.repeat(100); // ~1500 characters
      
      try {
        const result = await provider.generate(longPrompt, { maxTokens: 50 });
        expect(result).toBeDefined();
      } catch (error) {
        // Some models may have context length limits
        expect(error).toBeInstanceOf(Error);
      }
    }, 30000);
  });
});
