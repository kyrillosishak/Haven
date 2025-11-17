/**
 * Tests for PerformanceOptimizer
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { PerformanceOptimizer } from './PerformanceOptimizer';
import { IndexedDBStorage } from '../storage/IndexedDBStorage';
import type { VectorRecord } from '../storage/types';

describe('PerformanceOptimizer', () => {
  let optimizer: PerformanceOptimizer;
  let storage: IndexedDBStorage;

  beforeEach(async () => {
    optimizer = new PerformanceOptimizer({
      maxMemoryMB: 100,
      vectorCacheSize: 10 * 1024 * 1024,
      embeddingCacheSize: 5 * 1024 * 1024,
      indexCacheSize: 5 * 1024 * 1024,
      enableWorkers: false, // Disable workers for testing
      batchSize: 10,
      chunkSize: 100,
    });

    storage = new IndexedDBStorage({
      dbName: 'test-perf-optimizer',
      version: 1,
    });

    await storage.initialize();
    await optimizer.initialize(storage);
  });

  afterEach(async () => {
    await optimizer.dispose();
    await storage.destroy();
  });

  it('should initialize successfully', () => {
    expect(optimizer).toBeDefined();
  });

  it('should cache and retrieve vectors', async () => {
    const record: VectorRecord = {
      id: 'test1',
      vector: new Float32Array([1, 2, 3]),
      metadata: { content: 'test' },
      timestamp: Date.now(),
    };

    await storage.put(record);

    // First call should load from storage
    const result1 = await optimizer.getVector('test1', storage);
    expect(result1).toBeDefined();
    expect(result1?.id).toBe('test1');

    // Second call should use cache
    const result2 = await optimizer.getVector('test1', storage);
    expect(result2).toBeDefined();
    expect(result2?.id).toBe('test1');
  });

  it('should cache embeddings', () => {
    const embedding = new Float32Array([1, 2, 3, 4]);
    
    optimizer.cacheEmbedding('test text', embedding);
    
    const cached = optimizer.getCachedEmbedding('test text');
    expect(cached).toBeDefined();
    expect(cached?.length).toBe(4);
  });

  it('should cache index data', () => {
    const indexData = { type: 'test', data: [1, 2, 3] };
    
    optimizer.cacheIndex('test-index', indexData);
    
    const cached = optimizer.getCachedIndex('test-index');
    expect(cached).toBeDefined();
    expect(cached.type).toBe('test');
  });

  it('should track lazy loading state', () => {
    expect(optimizer.isIndexLoaded()).toBe(false);
    expect(optimizer.areModelsLoaded()).toBe(false);

    optimizer.markIndexLoaded();
    optimizer.markModelsLoaded();

    expect(optimizer.isIndexLoaded()).toBe(true);
    expect(optimizer.areModelsLoaded()).toBe(true);
  });

  it('should provide performance statistics', () => {
    const stats = optimizer.getStats();

    expect(stats).toHaveProperty('memory');
    expect(stats).toHaveProperty('caches');
    expect(stats.caches).toHaveProperty('vectors');
    expect(stats.caches).toHaveProperty('embeddings');
    expect(stats.caches).toHaveProperty('index');
  });

  it('should clear all caches', () => {
    optimizer.cacheEmbedding('test', new Float32Array([1, 2, 3]));
    optimizer.cacheIndex('test', { data: 'test' });

    optimizer.clearCaches();

    expect(optimizer.getCachedEmbedding('test')).toBeUndefined();
    expect(optimizer.getCachedIndex('test')).toBeUndefined();
  });

  it('should get vector batch with caching', async () => {
    const records: VectorRecord[] = [
      {
        id: 'test1',
        vector: new Float32Array([1, 2, 3]),
        metadata: { content: 'test1' },
        timestamp: Date.now(),
      },
      {
        id: 'test2',
        vector: new Float32Array([4, 5, 6]),
        metadata: { content: 'test2' },
        timestamp: Date.now(),
      },
    ];

    await storage.putBatch(records);

    const results = await optimizer.getVectorBatch(['test1', 'test2'], storage);
    expect(results.length).toBe(2);
    expect(results[0].id).toBe('test1');
    expect(results[1].id).toBe('test2');
  });
});
