/**
 * Integration tests for TransformersEmbedding with real models
 * 
 * These tests require:
 * - Real browser environment (not jsdom/happy-dom)
 * - Internet connection for model downloads
 * - WebGPU or WASM support
 * 
 * Run with: npm run test:integration
 * Skip with: SKIP_INTEGRATION=true npm test
 * 
 * Note: These tests are automatically skipped in CI environments
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { TransformersEmbedding } from './TransformersEmbedding';

// Skip integration tests in CI or when explicitly disabled
const skipIntegration = process.env.CI === 'true' || process.env.SKIP_INTEGRATION === 'true';

describe.skipIf(skipIntegration)('TransformersEmbedding (Integration)', () => {
  let embedding: TransformersEmbedding;

  beforeAll(async () => {
    // Initialize with a small, fast model for testing
    embedding = new TransformersEmbedding({
      model: 'Xenova/all-MiniLM-L6-v2',
      device: 'wasm', // Use WASM for better compatibility
      cache: true,
      quantized: true,
    });

    // This will download the real model from HuggingFace
    await embedding.initialize();
  }, 60000); // 60 second timeout for model download

  afterAll(async () => {
    if (embedding) {
      await embedding.dispose();
    }
  });

  describe('real model initialization', () => {
    it('should load real model from HuggingFace', () => {
      expect(embedding.getDimensions()).toBe(384);
    });

    it('should have valid embedding dimensions', () => {
      const dims = embedding.getDimensions();
      expect(dims).toBeGreaterThan(0);
      expect(Number.isInteger(dims)).toBe(true);
    });
  });

  describe('real embedding generation', () => {
    it('should generate real embeddings for text', async () => {
      const text = 'Machine learning is a subset of artificial intelligence';
      const result = await embedding.embed(text);

      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(384);
      
      // Check that embeddings are normalized
      let norm = 0;
      for (let i = 0; i < result.length; i++) {
        norm += result[i] * result[i];
      }
      norm = Math.sqrt(norm);
      expect(norm).toBeCloseTo(1.0, 5);
    });

    it('should generate consistent embeddings for same text', async () => {
      const text = 'Consistent embedding test';
      const result1 = await embedding.embed(text);
      const result2 = await embedding.embed(text);

      expect(result1.length).toBe(result2.length);
      
      // Real models should produce identical embeddings for identical input
      for (let i = 0; i < result1.length; i++) {
        expect(result1[i]).toBeCloseTo(result2[i], 6);
      }
    });

    it('should generate semantically meaningful embeddings', async () => {
      // Similar texts should have high similarity
      const text1 = 'The cat sits on the mat';
      const text2 = 'A cat is sitting on a mat';
      const text3 = 'Quantum physics is complex';

      const emb1 = await embedding.embed(text1);
      const emb2 = await embedding.embed(text2);
      const emb3 = await embedding.embed(text3);

      // Calculate cosine similarities
      const similarity12 = cosineSimilarity(emb1, emb2);
      const similarity13 = cosineSimilarity(emb1, emb3);

      // Similar texts should have higher similarity than unrelated texts
      expect(similarity12).toBeGreaterThan(similarity13);
      expect(similarity12).toBeGreaterThan(0.7); // High similarity threshold
    });

    it('should handle various text lengths', async () => {
      const texts = [
        'Short',
        'This is a medium length sentence for testing.',
        'This is a much longer text that contains multiple sentences. It should still be processed correctly by the embedding model. The model should handle various text lengths without issues.',
      ];

      for (const text of texts) {
        const result = await embedding.embed(text);
        expect(result).toBeInstanceOf(Float32Array);
        expect(result.length).toBe(384);
      }
    });

    it('should handle special characters and unicode', async () => {
      const texts = [
        'Hello, world! 🌍',
        'Café résumé naïve',
        '日本語のテキスト',
        'Emoji test: 😀 🎉 🚀',
      ];

      for (const text of texts) {
        const result = await embedding.embed(text);
        expect(result).toBeInstanceOf(Float32Array);
        expect(result.length).toBe(384);
      }
    });
  });

  describe('batch embedding with real model', () => {
    it('should generate embeddings for multiple texts', async () => {
      const texts = [
        'First document about machine learning',
        'Second document about deep learning',
        'Third document about neural networks',
      ];

      const results = await embedding.embedBatch(texts);

      expect(results).toHaveLength(3);
      results.forEach(result => {
        expect(result).toBeInstanceOf(Float32Array);
        expect(result.length).toBe(384);
      });
    });

    it('should maintain semantic relationships in batch', async () => {
      const texts = [
        'Dogs are loyal pets',
        'Cats are independent animals',
        'Python is a programming language',
      ];

      const results = await embedding.embedBatch(texts);

      // Calculate similarities
      const sim01 = cosineSimilarity(results[0], results[1]); // Dog vs Cat
      const sim02 = cosineSimilarity(results[0], results[2]); // Dog vs Python
      const sim12 = cosineSimilarity(results[1], results[2]); // Cat vs Python

      // Animal-related texts should be more similar to each other
      expect(sim01).toBeGreaterThan(sim02);
      expect(sim01).toBeGreaterThan(sim12);
    });
  });

  describe('performance with real model', () => {
    it('should embed text in reasonable time', async () => {
      const text = 'Performance test text';
      const startTime = Date.now();
      
      await embedding.embed(text);
      
      const duration = Date.now() - startTime;
      // Should complete in less than 5 seconds
      expect(duration).toBeLessThan(5000);
    });

    it('should handle batch embedding efficiently', async () => {
      const texts = Array(10).fill(0).map((_, i) => `Test text number ${i}`);
      const startTime = Date.now();
      
      await embedding.embedBatch(texts);
      
      const duration = Date.now() - startTime;
      // Batch should be reasonably fast (less than 10 seconds for 10 texts)
      expect(duration).toBeLessThan(10000);
    });
  });
});

/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dotProduct / (normA * normB);
}
