/**
 * Tests for TransformersEmbedding
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { TransformersEmbedding } from './TransformersEmbedding';
import { createMockPipeline } from '../test/mocks/transformers.js';

// Mock the @huggingface/transformers module
vi.mock('@huggingface/transformers', () => {
  const mockPipeline = vi.fn();
  return {
    pipeline: mockPipeline,
    env: {
      allowLocalModels: false,
      useBrowserCache: true,
      allowRemoteModels: true,
      cacheDir: './.cache/huggingface',
    },
  };
});

describe('TransformersEmbedding', () => {
  let embedding: TransformersEmbedding;

  beforeEach(async () => {
    // Reset all mocks before each test
    vi.clearAllMocks();
    
    // Import pipeline mock
    const { pipeline } = await import('@huggingface/transformers');
    
    // Setup default mock implementation
    (vi.mocked(pipeline) as any).mockResolvedValue(createMockPipeline({ dimensions: 384 }) as any);
    
    embedding = new TransformersEmbedding({
      model: 'Xenova/all-MiniLM-L6-v2',
      device: 'wasm',
      cache: false,
    });
  });

  afterEach(async () => {
    if (embedding) {
      await embedding.dispose();
    }
  });

  describe('initialization', () => {
    it('should initialize successfully with correct pipeline calls', async () => {
      const { pipeline } = await import('@huggingface/transformers');
      
      await embedding.initialize();
      
      // Verify pipeline was called with correct parameters
      expect(pipeline).toHaveBeenCalledWith(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2',
        expect.objectContaining({
          quantized: true,
        })
      );
      
      expect(embedding.getDimensions()).toBe(384);
    });

    it('should initialize with WebGPU device when specified', async () => {
      const { pipeline } = await import('@huggingface/transformers');
      
      const webgpuEmbedding = new TransformersEmbedding({
        model: 'test-model',
        device: 'webgpu',
        quantized: false,
      });
      
      await webgpuEmbedding.initialize();
      
      expect(pipeline).toHaveBeenCalledWith(
        'feature-extraction',
        'test-model',
        expect.objectContaining({
          device: 'webgpu',
          quantized: false,
        })
      );
      
      await webgpuEmbedding.dispose();
    });

    it('should throw error when getting dimensions before initialization', () => {
      expect(() => embedding.getDimensions()).toThrow('not initialized');
    });

    it('should not reinitialize if already initialized', async () => {
      const { pipeline } = await import('@huggingface/transformers');
      
      await embedding.initialize();
      const dims1 = embedding.getDimensions();
      
      await embedding.initialize();
      const dims2 = embedding.getDimensions();
      
      expect(dims1).toBe(dims2);
      // Pipeline should only be called once
      expect(pipeline).toHaveBeenCalledTimes(1);
    });

    it('should retry initialization on failure', async () => {
      const { pipeline } = await import('@huggingface/transformers');
      
      // Mock pipeline to fail twice then succeed
      (vi.mocked(pipeline) as any)
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce(createMockPipeline({ dimensions: 384 }) as any);
      
      const retryEmbedding = new TransformersEmbedding({
        model: 'test-model',
        maxRetries: 3,
        retryDelay: 10, // Short delay for testing
      });
      
      await retryEmbedding.initialize();
      
      // Should have been called 3 times (2 failures + 1 success)
      expect(pipeline).toHaveBeenCalledTimes(3);
      expect(retryEmbedding.getDimensions()).toBe(384);
      
      await retryEmbedding.dispose();
    });

    it('should fallback from WebGPU to WASM on failure', async () => {
      const { pipeline } = await import('@huggingface/transformers');
      
      // Mock pipeline to fail on WebGPU, succeed on WASM
      (vi.mocked(pipeline) as any)
        .mockRejectedValueOnce(new Error('WebGPU not supported'))
        .mockResolvedValueOnce(createMockPipeline({ dimensions: 384 }) as any);
      
      const webgpuEmbedding = new TransformersEmbedding({
        model: 'test-model',
        device: 'webgpu',
        maxRetries: 3,
        retryDelay: 10,
      });
      
      await webgpuEmbedding.initialize();
      
      // Should have been called twice (WebGPU fail + WASM success)
      expect(pipeline).toHaveBeenCalledTimes(2);
      
      // First call should be WebGPU
      expect(pipeline).toHaveBeenNthCalledWith(
        1,
        'feature-extraction',
        'test-model',
        expect.objectContaining({ device: 'webgpu' })
      );
      
      // Second call should be WASM (no device property)
      expect(pipeline).toHaveBeenNthCalledWith(
        2,
        'feature-extraction',
        'test-model',
        expect.not.objectContaining({ device: 'webgpu' })
      );
      
      expect(webgpuEmbedding.getDimensions()).toBe(384);
      
      await webgpuEmbedding.dispose();
    });

    it('should throw error after max retries exceeded', async () => {
      const { pipeline } = await import('@huggingface/transformers');
      
      // Mock pipeline to always fail
      (vi.mocked(pipeline) as any).mockRejectedValue(new Error('Persistent error'));
      
      const failingEmbedding = new TransformersEmbedding({
        model: 'test-model',
        maxRetries: 2,
        retryDelay: 10,
      });
      
      await expect(failingEmbedding.initialize()).rejects.toThrow(
        'Failed to initialize embedding model after 2 attempts'
      );
      
      // Should have been called maxRetries times
      expect(pipeline).toHaveBeenCalledTimes(2);
    });
  });

  describe('text embedding', () => {
    beforeEach(async () => {
      await embedding.initialize();
    });

    it('should generate embedding for single text', async () => {
      const text = 'Hello world';
      const result = await embedding.embed(text);
      
      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(embedding.getDimensions());
      expect(result.length).toBe(384);
    });

    it('should generate consistent embeddings for same text', async () => {
      const text = 'Test consistency';
      const result1 = await embedding.embed(text);
      const result2 = await embedding.embed(text);
      
      expect(result1.length).toBe(result2.length);
      
      // Mock generates deterministic embeddings, so they should be identical
      for (let i = 0; i < result1.length; i++) {
        expect(result1[i]).toBe(result2[i]);
      }
    });

    it('should generate different embeddings for different texts', async () => {
      const text1 = 'Machine learning';
      const text2 = 'Cooking recipes';
      
      const result1 = await embedding.embed(text1);
      const result2 = await embedding.embed(text2);
      
      expect(result1.length).toBe(result2.length);
      
      // Calculate cosine similarity
      let dotProduct = 0;
      for (let i = 0; i < result1.length; i++) {
        dotProduct += result1[i] * result2[i];
      }
      
      // Different texts should produce different embeddings
      expect(dotProduct).not.toBe(1.0);
    });

    it('should generate normalized embeddings', async () => {
      const text = 'Test normalization';
      const result = await embedding.embed(text);
      
      // Calculate L2 norm
      let norm = 0;
      for (let i = 0; i < result.length; i++) {
        norm += result[i] * result[i];
      }
      norm = Math.sqrt(norm);
      
      // Normalized embeddings should have L2 norm of 1
      expect(norm).toBeCloseTo(1.0, 5);
    });

    it('should throw error when embedding before initialization', async () => {
      const uninitializedEmbedding = new TransformersEmbedding({
        model: 'Xenova/all-MiniLM-L6-v2',
      });
      
      await expect(uninitializedEmbedding.embed('test')).rejects.toThrow('not initialized');
    });

    it('should handle embedding generation errors', async () => {
      await embedding.initialize();
      
      // Mock the pipeline to throw an error
      const { pipeline } = await import('@huggingface/transformers');
      const mockPipelineInstance = createMockPipeline({ dimensions: 384 });
      const originalFn = mockPipelineInstance;
      
      // Replace the mock to throw an error
      (vi.mocked(pipeline) as any).mockResolvedValue(
        (async () => {
          throw new Error('Embedding generation failed');
        }) as any
      );
      
      // Create new instance with failing pipeline
      const failingEmbedding = new TransformersEmbedding({
        model: 'test-model',
      });
      
      await expect(failingEmbedding.initialize()).rejects.toThrow();
    });
  });

  describe('batch embedding', () => {
    beforeEach(async () => {
      await embedding.initialize();
    });

    it('should generate embeddings for multiple texts', async () => {
      const texts = ['First text', 'Second text', 'Third text'];
      const results = await embedding.embedBatch(texts);
      
      expect(results).toHaveLength(3);
      results.forEach(result => {
        expect(result).toBeInstanceOf(Float32Array);
        expect(result.length).toBe(embedding.getDimensions());
      });
    });

    it('should generate different embeddings for different texts in batch', async () => {
      const texts = ['First text', 'Second text'];
      const results = await embedding.embedBatch(texts);
      
      expect(results).toHaveLength(2);
      
      // Embeddings should be different
      let allSame = true;
      for (let i = 0; i < results[0].length; i++) {
        if (results[0][i] !== results[1][i]) {
          allSame = false;
          break;
        }
      }
      expect(allSame).toBe(false);
    });

    it('should return empty array for empty input', async () => {
      const results = await embedding.embedBatch([]);
      expect(results).toHaveLength(0);
    });

    it('should handle single text in batch', async () => {
      const texts = ['Single text'];
      const results = await embedding.embedBatch(texts);
      
      expect(results).toHaveLength(1);
      expect(results[0]).toBeInstanceOf(Float32Array);
    });

    it('should throw error when batch embedding before initialization', async () => {
      const uninitializedEmbedding = new TransformersEmbedding({
        model: 'test-model',
      });
      
      await expect(uninitializedEmbedding.embedBatch(['test'])).rejects.toThrow('not initialized');
    });
  });

  describe('disposal', () => {
    it('should dispose resources', async () => {
      await embedding.initialize();
      expect(embedding.getDimensions()).toBe(384);
      
      await embedding.dispose();
      
      expect(() => embedding.getDimensions()).toThrow('not initialized');
    });

    it('should allow reinitialization after disposal', async () => {
      const { pipeline } = await import('@huggingface/transformers');
      
      await embedding.initialize();
      const dims1 = embedding.getDimensions();
      
      await embedding.dispose();
      
      // Reset mock for reinitialization
      (vi.mocked(pipeline) as any).mockResolvedValue(createMockPipeline({ dimensions: 384 }) as any);
      
      await embedding.initialize();
      
      const dims2 = embedding.getDimensions();
      expect(dims2).toBe(dims1);
    });

    it('should not allow embedding after disposal', async () => {
      await embedding.initialize();
      await embedding.dispose();
      
      await expect(embedding.embed('test')).rejects.toThrow('not initialized');
    });
  });

  describe('error handling', () => {
    it('should handle pipeline initialization errors gracefully', async () => {
      const { pipeline } = await import('@huggingface/transformers');
      
      (vi.mocked(pipeline) as any).mockRejectedValue(new Error('Model not found'));
      
      const errorEmbedding = new TransformersEmbedding({
        model: 'invalid-model',
        maxRetries: 1,
        retryDelay: 10,
      });
      
      await expect(errorEmbedding.initialize()).rejects.toThrow(
        'Failed to initialize embedding model'
      );
    });

    it('should handle unexpected pipeline output format', async () => {
      const { pipeline } = await import('@huggingface/transformers');
      
      // Mock pipeline with unexpected output format
      (vi.mocked(pipeline) as any).mockResolvedValue(
        (async () => ({
          // Missing data and tolist properties
          unexpected: 'format',
        })) as any
      );
      
      const badEmbedding = new TransformersEmbedding({
        model: 'test-model',
      });
      
      await expect(badEmbedding.initialize()).rejects.toThrow();
    });

    it('should validate initialization state before operations', async () => {
      const uninitializedEmbedding = new TransformersEmbedding({
        model: 'test-model',
      });
      
      // All operations should fail before initialization
      expect(() => uninitializedEmbedding.getDimensions()).toThrow('not initialized');
      await expect(uninitializedEmbedding.embed('test')).rejects.toThrow('not initialized');
      await expect(uninitializedEmbedding.embedBatch(['test'])).rejects.toThrow('not initialized');
    });
  });

  describe('configuration', () => {
    it('should respect cache configuration', async () => {
      const { pipeline, env } = await import('@huggingface/transformers');
      
      const cachedEmbedding = new TransformersEmbedding({
        model: 'test-model',
        cache: true,
      });
      
      await cachedEmbedding.initialize();
      
      // Verify env was configured for caching
      expect(env.useBrowserCache).toBe(true);
      
      await cachedEmbedding.dispose();
    });

    it('should respect quantization configuration', async () => {
      const { pipeline } = await import('@huggingface/transformers');
      
      const quantizedEmbedding = new TransformersEmbedding({
        model: 'test-model',
        quantized: false,
      });
      
      await quantizedEmbedding.initialize();
      
      expect(pipeline).toHaveBeenCalledWith(
        'feature-extraction',
        'test-model',
        expect.objectContaining({ quantized: false })
      );
      
      await quantizedEmbedding.dispose();
    });

    it('should use default configuration values', async () => {
      const { pipeline } = await import('@huggingface/transformers');
      
      const defaultEmbedding = new TransformersEmbedding({
        model: 'test-model',
      });
      
      await defaultEmbedding.initialize();
      
      // Should use default values: device=wasm, cache=true, quantized=true
      expect(pipeline).toHaveBeenCalledWith(
        'feature-extraction',
        'test-model',
        expect.objectContaining({ quantized: true })
      );
      
      await defaultEmbedding.dispose();
    });
  });
});
