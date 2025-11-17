/**
 * IndexManager tests
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { IndexManager } from './IndexManager';
import { IndexedDBStorage } from '../storage/IndexedDBStorage';
import type { VectorRecord } from '../storage/types';
import { DimensionMismatchError, IndexCorruptedError } from '../errors';

describe('IndexManager', () => {
  let storage: IndexedDBStorage;
  let indexManager: IndexManager;
  const dimensions = 3;

  beforeEach(async () => {
    storage = new IndexedDBStorage({
      dbName: `test-index-${Date.now()}`,
      version: 1,
    });
    await storage.initialize();

    indexManager = new IndexManager({
      dimensions,
      metric: 'cosine',
      storage,
    });
    await indexManager.initialize();
  });

  describe('initialization', () => {
    it('should initialize successfully', async () => {
      const stats = indexManager.getStats();
      expect(stats.vectorCount).toBe(0);
      expect(stats.dimensions).toBe(dimensions);
      expect(stats.indexType).toBe('voy');
    });

    it('should load existing index from storage', async () => {
      // Add a vector
      const vector: VectorRecord = {
        id: 'test-1',
        vector: new Float32Array([1, 0, 0]),
        metadata: { content: 'test' },
        timestamp: Date.now(),
      };
      await storage.put(vector);
      await indexManager.add(vector);

      // Create new index manager with same storage
      const newIndexManager = new IndexManager({
        dimensions,
        metric: 'cosine',
        storage,
      });
      await newIndexManager.initialize();

      const stats = newIndexManager.getStats();
      expect(stats.vectorCount).toBe(1);
    });
  });

  describe('build', () => {
    it('should build index from vectors', async () => {
      const vectors: VectorRecord[] = [
        {
          id: 'v1',
          vector: new Float32Array([1, 0, 0]),
          metadata: { content: 'first' },
          timestamp: Date.now(),
        },
        {
          id: 'v2',
          vector: new Float32Array([0, 1, 0]),
          metadata: { content: 'second' },
          timestamp: Date.now(),
        },
      ];

      await indexManager.build(vectors);

      const stats = indexManager.getStats();
      expect(stats.vectorCount).toBe(2);
    });

    it('should handle empty vector array', async () => {
      await indexManager.build([]);
      const stats = indexManager.getStats();
      expect(stats.vectorCount).toBe(0);
    });

    it('should throw on dimension mismatch', async () => {
      const vectors: VectorRecord[] = [
        {
          id: 'v1',
          vector: new Float32Array([1, 0]),
          metadata: { content: 'wrong dimensions' },
          timestamp: Date.now(),
        },
      ];

      await expect(indexManager.build(vectors)).rejects.toThrow(DimensionMismatchError);
    });
  });

  describe('add', () => {
    it('should add single vector', async () => {
      const vector: VectorRecord = {
        id: 'v1',
        vector: new Float32Array([1, 0, 0]),
        metadata: { content: 'test' },
        timestamp: Date.now(),
      };

      await storage.put(vector);
      await indexManager.add(vector);

      const stats = indexManager.getStats();
      expect(stats.vectorCount).toBe(1);
    });

    it('should throw on dimension mismatch', async () => {
      const vector: VectorRecord = {
        id: 'v1',
        vector: new Float32Array([1, 0]),
        metadata: { content: 'wrong' },
        timestamp: Date.now(),
      };

      await expect(indexManager.add(vector)).rejects.toThrow(DimensionMismatchError);
    });
  });

  describe('addBatch', () => {
    it('should add multiple vectors', async () => {
      const vectors: VectorRecord[] = [
        {
          id: 'v1',
          vector: new Float32Array([1, 0, 0]),
          metadata: { content: 'first' },
          timestamp: Date.now(),
        },
        {
          id: 'v2',
          vector: new Float32Array([0, 1, 0]),
          metadata: { content: 'second' },
          timestamp: Date.now(),
        },
      ];

      await storage.putBatch(vectors);
      await indexManager.addBatch(vectors);

      const stats = indexManager.getStats();
      expect(stats.vectorCount).toBe(2);
    });

    it('should handle empty array', async () => {
      await indexManager.addBatch([]);
      const stats = indexManager.getStats();
      expect(stats.vectorCount).toBe(0);
    });
  });

  describe('remove', () => {
    it('should remove vector by id', async () => {
      const vectors: VectorRecord[] = [
        {
          id: 'v1',
          vector: new Float32Array([1, 0, 0]),
          metadata: { content: 'first' },
          timestamp: Date.now(),
        },
        {
          id: 'v2',
          vector: new Float32Array([0, 1, 0]),
          metadata: { content: 'second' },
          timestamp: Date.now(),
        },
      ];

      await storage.putBatch(vectors);
      await indexManager.build(vectors);

      await indexManager.remove('v1');

      const stats = indexManager.getStats();
      expect(stats.vectorCount).toBe(1);
    });
  });

  describe('search', () => {
    beforeEach(async () => {
      const vectors: VectorRecord[] = [
        {
          id: 'v1',
          vector: new Float32Array([1, 0, 0]),
          metadata: { content: 'first', category: 'A' },
          timestamp: Date.now(),
        },
        {
          id: 'v2',
          vector: new Float32Array([0, 1, 0]),
          metadata: { content: 'second', category: 'B' },
          timestamp: Date.now(),
        },
        {
          id: 'v3',
          vector: new Float32Array([0, 0, 1]),
          metadata: { content: 'third', category: 'A' },
          timestamp: Date.now(),
        },
      ];

      await storage.putBatch(vectors);
      await indexManager.build(vectors);
    });

    it('should find nearest neighbors', async () => {
      const query = new Float32Array([1, 0, 0]);
      const results = await indexManager.search(query, 2);

      expect(results.length).toBeGreaterThan(0);
      expect(results.length).toBeLessThanOrEqual(2);
      expect(results[0].id).toBe('v1');
    });

    it('should apply metadata filters', async () => {
      const query = new Float32Array([1, 0, 0]);
      const results = await indexManager.search(query, 5, {
        field: 'category',
        operator: 'eq',
        value: 'A',
      });

      expect(results.length).toBeGreaterThan(0);
      for (const result of results) {
        expect(result.metadata.category).toBe('A');
      }
    });

    it('should apply compound AND filters', async () => {
      // Add more vectors with varied metadata
      const moreVectors: VectorRecord[] = [
        {
          id: 'v4',
          vector: new Float32Array([0.9, 0.1, 0]),
          metadata: { content: 'fourth', category: 'A', score: 10 },
          timestamp: Date.now(),
        },
        {
          id: 'v5',
          vector: new Float32Array([0.8, 0.2, 0]),
          metadata: { content: 'fifth', category: 'A', score: 5 },
          timestamp: Date.now(),
        },
      ];
      await storage.putBatch(moreVectors);
      await indexManager.addBatch(moreVectors);

      const query = new Float32Array([1, 0, 0]);
      const results = await indexManager.search(query, 10, {
        operator: 'and',
        filters: [
          { field: 'category', operator: 'eq', value: 'A' },
          { field: 'score', operator: 'gte', value: 10 },
        ],
      });

      expect(results.length).toBeGreaterThan(0);
      for (const result of results) {
        expect(result.metadata.category).toBe('A');
        expect(result.metadata.score).toBeGreaterThanOrEqual(10);
      }
    });

    it('should apply compound OR filters', async () => {
      const query = new Float32Array([1, 0, 0]);
      const results = await indexManager.search(query, 10, {
        operator: 'or',
        filters: [
          { field: 'category', operator: 'eq', value: 'A' },
          { field: 'category', operator: 'eq', value: 'B' },
        ],
      });

      expect(results.length).toBeGreaterThan(0);
      for (const result of results) {
        expect(['A', 'B']).toContain(result.metadata.category);
      }
    });

    it('should apply nested compound filters', async () => {
      // Add vectors with tags
      const taggedVectors: VectorRecord[] = [
        {
          id: 'v6',
          vector: new Float32Array([0.7, 0.3, 0]),
          metadata: { content: 'sixth', category: 'A', tags: ['important', 'urgent'] },
          timestamp: Date.now(),
        },
        {
          id: 'v7',
          vector: new Float32Array([0.6, 0.4, 0]),
          metadata: { content: 'seventh', category: 'B', tags: ['important'] },
          timestamp: Date.now(),
        },
      ];
      await storage.putBatch(taggedVectors);
      await indexManager.addBatch(taggedVectors);

      const query = new Float32Array([1, 0, 0]);
      const results = await indexManager.search(query, 10, {
        operator: 'and',
        filters: [
          {
            operator: 'or',
            filters: [
              { field: 'category', operator: 'eq', value: 'A' },
              { field: 'category', operator: 'eq', value: 'B' },
            ],
          },
          { field: 'tags', operator: 'contains', value: 'important' },
        ],
      });

      expect(results.length).toBeGreaterThan(0);
      for (const result of results) {
        expect(['A', 'B']).toContain(result.metadata.category);
        expect(result.metadata.tags).toContain('important');
      }
    });

    it('should return empty array for empty index', async () => {
      await indexManager.clear();
      const query = new Float32Array([1, 0, 0]);
      const results = await indexManager.search(query, 5);

      expect(results).toEqual([]);
    });

    it('should throw on dimension mismatch', async () => {
      const query = new Float32Array([1, 0]);
      await expect(indexManager.search(query, 5)).rejects.toThrow(DimensionMismatchError);
    });
  });

  describe('serialization', () => {
    it('should serialize and deserialize index', async () => {
      const vectors: VectorRecord[] = [
        {
          id: 'v1',
          vector: new Float32Array([1, 0, 0]),
          metadata: { content: 'test' },
          timestamp: Date.now(),
        },
      ];

      await indexManager.build(vectors);

      const serialized = await indexManager.serialize();
      expect(serialized).toBeTruthy();
      expect(typeof serialized).toBe('string');

      const newIndexManager = new IndexManager({
        dimensions,
        metric: 'cosine',
        storage,
      });
      await newIndexManager.deserialize(serialized);

      const stats = newIndexManager.getStats();
      expect(stats.vectorCount).toBe(1);
    });

    it('should throw on corrupted data', async () => {
      const newIndexManager = new IndexManager({
        dimensions,
        metric: 'cosine',
        storage,
      });

      await expect(newIndexManager.deserialize('invalid json')).rejects.toThrow(
        IndexCorruptedError
      );
    });

    it('should throw on dimension mismatch in deserialization', async () => {
      const vectors: VectorRecord[] = [
        {
          id: 'v1',
          vector: new Float32Array([1, 0, 0]),
          metadata: { content: 'test' },
          timestamp: Date.now(),
        },
      ];

      await indexManager.build(vectors);
      const serialized = await indexManager.serialize();

      const newIndexManager = new IndexManager({
        dimensions: 5, // Different dimensions
        metric: 'cosine',
        storage,
      });

      await expect(newIndexManager.deserialize(serialized)).rejects.toThrow(
        DimensionMismatchError
      );
    });
  });

  describe('clear', () => {
    it('should clear all vectors from index', async () => {
      const vectors: VectorRecord[] = [
        {
          id: 'v1',
          vector: new Float32Array([1, 0, 0]),
          metadata: { content: 'test' },
          timestamp: Date.now(),
        },
      ];

      await indexManager.build(vectors);
      await indexManager.clear();

      const stats = indexManager.getStats();
      expect(stats.vectorCount).toBe(0);
    });
  });

  describe('getStats', () => {
    it('should return accurate statistics', async () => {
      const vectors: VectorRecord[] = [
        {
          id: 'v1',
          vector: new Float32Array([1, 0, 0]),
          metadata: { content: 'test' },
          timestamp: Date.now(),
        },
      ];

      await indexManager.build(vectors);

      const stats = indexManager.getStats();
      expect(stats.vectorCount).toBe(1);
      expect(stats.dimensions).toBe(dimensions);
      expect(stats.indexType).toBe('voy');
      expect(stats.memoryUsage).toBeGreaterThan(0);
      expect(stats.lastUpdated).toBeGreaterThan(0);
    });
  });
});
