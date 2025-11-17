/**
 * Tests for IndexedDB Storage Manager
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { IndexedDBStorage } from './IndexedDBStorage';
import { VectorRecord, MetadataFilter, CompoundFilter } from './types';
import { StorageQuotaError, VectorDBError, IndexCorruptedError } from '../errors';

describe('IndexedDBStorage', () => {
  let storage: IndexedDBStorage;
  const testDbName = 'test-vectordb';

  beforeEach(async () => {
    storage = new IndexedDBStorage({ dbName: testDbName, version: 1 });
    await storage.initialize();
  });

  afterEach(async () => {
    await storage.destroy();
  });

  describe('Initialization', () => {
    it('should initialize database successfully', async () => {
      const newStorage = new IndexedDBStorage({ dbName: 'test-init', version: 1 });
      await expect(newStorage.initialize()).resolves.not.toThrow();
      await newStorage.destroy();
    });
  });

  describe('CRUD Operations', () => {
    it('should store and retrieve a vector record', async () => {
      const record: VectorRecord = {
        id: 'test-1',
        vector: new Float32Array([1, 2, 3, 4]),
        metadata: { content: 'test content', tags: ['test'] },
        timestamp: Date.now()
      };

      await storage.put(record);
      const retrieved = await storage.get('test-1');

      expect(retrieved).not.toBeNull();
      expect(retrieved!.id).toBe(record.id);
      expect(Array.from(retrieved!.vector)).toEqual(Array.from(record.vector));
      expect(retrieved!.metadata).toEqual(record.metadata);
    });

    it('should return null for non-existent record', async () => {
      const retrieved = await storage.get('non-existent');
      expect(retrieved).toBeNull();
    });

    it('should update existing record', async () => {
      const record: VectorRecord = {
        id: 'test-1',
        vector: new Float32Array([1, 2, 3]),
        metadata: { content: 'original' },
        timestamp: Date.now()
      };

      await storage.put(record);

      const updated: VectorRecord = {
        ...record,
        metadata: { content: 'updated' }
      };

      await storage.put(updated);
      const retrieved = await storage.get('test-1');

      expect(retrieved!.metadata.content).toBe('updated');
    });

    it('should delete a record', async () => {
      const record: VectorRecord = {
        id: 'test-1',
        vector: new Float32Array([1, 2, 3]),
        metadata: { content: 'test' },
        timestamp: Date.now()
      };

      await storage.put(record);
      const deleted = await storage.delete('test-1');
      expect(deleted).toBe(true);

      const retrieved = await storage.get('test-1');
      expect(retrieved).toBeNull();
    });

    it('should return false when deleting non-existent record', async () => {
      const deleted = await storage.delete('non-existent');
      expect(deleted).toBe(false);
    });

    it('should clear all records', async () => {
      const records: VectorRecord[] = [
        {
          id: 'test-1',
          vector: new Float32Array([1, 2, 3]),
          metadata: { content: 'test1' },
          timestamp: Date.now()
        },
        {
          id: 'test-2',
          vector: new Float32Array([4, 5, 6]),
          metadata: { content: 'test2' },
          timestamp: Date.now()
        }
      ];

      await storage.putBatch(records);
      await storage.clear();

      const count = await storage.count();
      expect(count).toBe(0);
    });
  });

  describe('Batch Operations', () => {
    it('should store multiple records in batch', async () => {
      const records: VectorRecord[] = [
        {
          id: 'batch-1',
          vector: new Float32Array([1, 2, 3]),
          metadata: { content: 'batch1' },
          timestamp: Date.now()
        },
        {
          id: 'batch-2',
          vector: new Float32Array([4, 5, 6]),
          metadata: { content: 'batch2' },
          timestamp: Date.now()
        },
        {
          id: 'batch-3',
          vector: new Float32Array([7, 8, 9]),
          metadata: { content: 'batch3' },
          timestamp: Date.now()
        }
      ];

      await storage.putBatch(records);

      const retrieved1 = await storage.get('batch-1');
      const retrieved2 = await storage.get('batch-2');
      const retrieved3 = await storage.get('batch-3');

      expect(retrieved1).not.toBeNull();
      expect(retrieved2).not.toBeNull();
      expect(retrieved3).not.toBeNull();
    });

    it('should retrieve multiple records in batch', async () => {
      const records: VectorRecord[] = [
        {
          id: 'batch-1',
          vector: new Float32Array([1, 2, 3]),
          metadata: { content: 'batch1' },
          timestamp: Date.now()
        },
        {
          id: 'batch-2',
          vector: new Float32Array([4, 5, 6]),
          metadata: { content: 'batch2' },
          timestamp: Date.now()
        }
      ];

      await storage.putBatch(records);
      const retrieved = await storage.getBatch(['batch-1', 'batch-2']);

      expect(retrieved).toHaveLength(2);
      expect(retrieved.map(r => r.id)).toContain('batch-1');
      expect(retrieved.map(r => r.id)).toContain('batch-2');
    });

    it('should handle empty batch operations', async () => {
      await expect(storage.putBatch([])).resolves.not.toThrow();
      const retrieved = await storage.getBatch([]);
      expect(retrieved).toHaveLength(0);
    });
  });

  describe('Metadata Filtering', () => {
    beforeEach(async () => {
      const records: VectorRecord[] = [
        {
          id: 'filter-1',
          vector: new Float32Array([1, 2, 3]),
          metadata: { type: 'document', score: 10, tags: ['important', 'urgent'] },
          timestamp: Date.now()
        },
        {
          id: 'filter-2',
          vector: new Float32Array([4, 5, 6]),
          metadata: { type: 'image', score: 20, tags: ['important'] },
          timestamp: Date.now()
        },
        {
          id: 'filter-3',
          vector: new Float32Array([7, 8, 9]),
          metadata: { type: 'document', score: 15, tags: ['urgent'] },
          timestamp: Date.now()
        }
      ];

      await storage.putBatch(records);
    });

    it('should filter by equality', async () => {
      const filter: MetadataFilter = {
        field: 'type',
        operator: 'eq',
        value: 'document'
      };

      const results = await storage.filter(filter);
      expect(results).toHaveLength(2);
      expect(results.every(r => r.metadata.type === 'document')).toBe(true);
    });

    it('should filter by inequality', async () => {
      const filter: MetadataFilter = {
        field: 'type',
        operator: 'ne',
        value: 'document'
      };

      const results = await storage.filter(filter);
      expect(results).toHaveLength(1);
      expect(results[0].metadata.type).toBe('image');
    });

    it('should filter by greater than', async () => {
      const filter: MetadataFilter = {
        field: 'score',
        operator: 'gt',
        value: 10
      };

      const results = await storage.filter(filter);
      expect(results).toHaveLength(2);
      expect(results.every(r => r.metadata.score > 10)).toBe(true);
    });

    it('should filter by less than or equal', async () => {
      const filter: MetadataFilter = {
        field: 'score',
        operator: 'lte',
        value: 15
      };

      const results = await storage.filter(filter);
      expect(results).toHaveLength(2);
      expect(results.every(r => r.metadata.score <= 15)).toBe(true);
    });

    it('should filter by in operator', async () => {
      const filter: MetadataFilter = {
        field: 'type',
        operator: 'in',
        value: ['document', 'video']
      };

      const results = await storage.filter(filter);
      expect(results).toHaveLength(2);
    });

    it('should filter by contains operator on arrays', async () => {
      const filter: MetadataFilter = {
        field: 'tags',
        operator: 'contains',
        value: 'urgent'
      };

      const results = await storage.filter(filter);
      expect(results).toHaveLength(2);
      expect(results.every(r => r.metadata.tags.includes('urgent'))).toBe(true);
    });

    it('should return empty array when no matches', async () => {
      const filter: MetadataFilter = {
        field: 'type',
        operator: 'eq',
        value: 'nonexistent'
      };

      const results = await storage.filter(filter);
      expect(results).toHaveLength(0);
    });

    it('should apply compound AND filters', async () => {
      const results = await storage.filter({
        operator: 'and',
        filters: [
          { field: 'type', operator: 'eq', value: 'document' },
          { field: 'score', operator: 'gte', value: 15 },
        ],
      });

      expect(results).toHaveLength(1);
      expect(results[0].id).toBe('filter-3');
      expect(results[0].metadata.type).toBe('document');
      expect(results[0].metadata.score).toBeGreaterThanOrEqual(15);
    });

    it('should apply compound OR filters', async () => {
      const results = await storage.filter({
        operator: 'or',
        filters: [
          { field: 'type', operator: 'eq', value: 'image' },
          { field: 'score', operator: 'eq', value: 10 },
        ],
      });

      expect(results).toHaveLength(2);
      const ids = results.map(r => r.id);
      expect(ids).toContain('filter-1');
      expect(ids).toContain('filter-2');
    });

    it('should apply nested compound filters', async () => {
      const results = await storage.filter({
        operator: 'and',
        filters: [
          {
            operator: 'or',
            filters: [
              { field: 'type', operator: 'eq', value: 'document' },
              { field: 'type', operator: 'eq', value: 'image' },
            ],
          },
          { field: 'tags', operator: 'contains', value: 'important' },
        ],
      });

      expect(results).toHaveLength(2);
      for (const result of results) {
        expect(['document', 'image']).toContain(result.metadata.type);
        expect(result.metadata.tags).toContain('important');
      }
    });

    it('should handle empty compound filter', async () => {
      const results = await storage.filter({
        operator: 'and',
        filters: [],
      });

      expect(results).toHaveLength(3);
    });
  });

  describe('Count', () => {
    it('should return correct count', async () => {
      const records: VectorRecord[] = [
        {
          id: 'count-1',
          vector: new Float32Array([1, 2, 3]),
          metadata: { content: 'test1' },
          timestamp: Date.now()
        },
        {
          id: 'count-2',
          vector: new Float32Array([4, 5, 6]),
          metadata: { content: 'test2' },
          timestamp: Date.now()
        }
      ];

      await storage.putBatch(records);
      const count = await storage.count();
      expect(count).toBe(2);
    });

    it('should return zero for empty database', async () => {
      const count = await storage.count();
      expect(count).toBe(0);
    });
  });

  describe('Index Persistence', () => {
    it('should save and load index', async () => {
      const indexData = JSON.stringify({ test: 'index data' });

      await storage.saveIndex(indexData);
      const loaded = await storage.loadIndex();

      expect(loaded).toBe(indexData);
    });

    it('should return null when no index exists', async () => {
      const loaded = await storage.loadIndex();
      expect(loaded).toBeNull();
    });

    it('should overwrite existing index', async () => {
      const indexData1 = JSON.stringify({ version: 1 });
      const indexData2 = JSON.stringify({ version: 2 });

      await storage.saveIndex(indexData1);
      await storage.saveIndex(indexData2);
      const loaded = await storage.loadIndex();

      expect(loaded).toBe(indexData2);
    });
  });

  describe('Error Handling', () => {
    it('should throw error when not initialized', async () => {
      const uninitializedStorage = new IndexedDBStorage({ dbName: 'uninit' });

      await expect(uninitializedStorage.get('test')).rejects.toThrow(VectorDBError);
      await expect(uninitializedStorage.put({
        id: 'test',
        vector: new Float32Array([1, 2, 3]),
        metadata: {},
        timestamp: Date.now()
      })).rejects.toThrow(VectorDBError);
    });
  });
});
