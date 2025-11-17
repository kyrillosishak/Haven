/**
 * Performance Optimizer - Main integration point for all performance optimizations
 * Coordinates caching, memory management, worker pools, and batch operations
 */

import { LRUCache } from './LRUCache';
import { MemoryManager } from './MemoryManager';
import { WorkerPool } from './WorkerPool';
import { ProgressiveLoader } from './ProgressiveLoader';
import { BatchOptimizer } from './BatchOptimizer';
import type { VectorRecord, StorageManager } from '../storage/types';

export interface PerformanceConfig {
  // Memory configuration
  maxMemoryMB?: number;
  evictionThreshold?: number;
  
  // Cache configuration
  vectorCacheSize?: number;
  embeddingCacheSize?: number;
  indexCacheSize?: number;
  
  // Worker configuration
  enableWorkers?: boolean;
  maxWorkers?: number;
  
  // Batch configuration
  batchSize?: number;
  batchWaitTime?: number;
  
  // Progressive loading
  chunkSize?: number;
  
  // Lazy loading
  lazyLoadIndex?: boolean;
  lazyLoadModels?: boolean;
}

/**
 * Coordinates all performance optimizations
 */
export class PerformanceOptimizer {
  private config: Required<PerformanceConfig>;
  
  // Caches
  public vectorCache: LRUCache<VectorRecord>;
  public embeddingCache: LRUCache<Float32Array>;
  public indexCache: LRUCache<any>;
  
  // Managers
  public memoryManager: MemoryManager;
  public workerPool: WorkerPool | null = null;
  public progressiveLoader: ProgressiveLoader;
  public batchOptimizer: BatchOptimizer | null = null;
  
  // State
  private initialized: boolean = false;
  private indexLoaded: boolean = false;
  private modelsLoaded: boolean = false;

  constructor(config: PerformanceConfig = {}) {
    this.config = {
      maxMemoryMB: 500,
      evictionThreshold: 0.9,
      vectorCacheSize: 100 * 1024 * 1024, // 100MB
      embeddingCacheSize: 50 * 1024 * 1024, // 50MB
      indexCacheSize: 100 * 1024 * 1024, // 100MB
      enableWorkers: true,
      maxWorkers: navigator.hardwareConcurrency || 4,
      batchSize: 100,
      batchWaitTime: 100,
      chunkSize: 1000,
      lazyLoadIndex: true,
      lazyLoadModels: true,
      ...config,
    };

    // Initialize caches
    this.vectorCache = new LRUCache<VectorRecord>({
      maxSize: this.config.vectorCacheSize,
      maxEntries: 10000,
      onEvict: (key) => {
        console.debug(`Evicted vector from cache: ${key}`);
      },
    });

    this.embeddingCache = new LRUCache<Float32Array>({
      maxSize: this.config.embeddingCacheSize,
      maxEntries: 5000,
      onEvict: (key) => {
        console.debug(`Evicted embedding from cache: ${key}`);
      },
    });

    this.indexCache = new LRUCache<any>({
      maxSize: this.config.indexCacheSize,
      maxEntries: 100,
      onEvict: (key) => {
        console.debug(`Evicted index data from cache: ${key}`);
      },
    });

    // Initialize memory manager
    this.memoryManager = new MemoryManager({
      maxMemoryMB: this.config.maxMemoryMB,
      evictionThreshold: this.config.evictionThreshold,
      checkInterval: 30000,
    });

    // Register caches with memory manager
    this.memoryManager.registerCache('vectors', this.vectorCache);
    this.memoryManager.registerCache('embeddings', this.embeddingCache);
    this.memoryManager.registerCache('index', this.indexCache);

    // Initialize progressive loader
    this.progressiveLoader = new ProgressiveLoader({
      chunkSize: this.config.chunkSize,
    });

    // Initialize worker pool if enabled
    if (this.config.enableWorkers) {
      this.workerPool = new WorkerPool({
        maxWorkers: this.config.maxWorkers,
      });
    }
  }

  /**
   * Initialize the performance optimizer
   */
  async initialize(storage?: StorageManager): Promise<void> {
    if (this.initialized) {
      return;
    }

    // Initialize batch optimizer if storage provided
    if (storage) {
      this.batchOptimizer = new BatchOptimizer(storage, {
        maxBatchSize: this.config.batchSize,
        maxWaitTime: this.config.batchWaitTime,
        autoFlush: true,
      });
    }

    // Start memory monitoring
    this.memoryManager.startMonitoring();

    this.initialized = true;
  }

  /**
   * Get a vector from cache or storage
   */
  async getVector(
    id: string,
    storage: StorageManager
  ): Promise<VectorRecord | null> {
    // Check cache first
    const cached = this.vectorCache.get(id);
    if (cached) {
      return cached;
    }

    // Load from storage
    const record = await storage.get(id);
    if (record) {
      // Add to cache
      const size = this.estimateVectorSize(record);
      this.vectorCache.set(id, record, size);
    }

    return record;
  }

  /**
   * Get multiple vectors with caching
   */
  async getVectorBatch(
    ids: string[],
    storage: StorageManager
  ): Promise<VectorRecord[]> {
    const results: VectorRecord[] = [];
    const uncachedIds: string[] = [];

    // Check cache for each ID
    for (const id of ids) {
      const cached = this.vectorCache.get(id);
      if (cached) {
        results.push(cached);
      } else {
        uncachedIds.push(id);
      }
    }

    // Load uncached vectors from storage
    if (uncachedIds.length > 0) {
      const records = await storage.getBatch(uncachedIds);
      
      for (const record of records) {
        results.push(record);
        
        // Add to cache
        const size = this.estimateVectorSize(record);
        this.vectorCache.set(record.id, record, size);
      }
    }

    return results;
  }

  /**
   * Cache an embedding
   */
  cacheEmbedding(text: string, embedding: Float32Array): void {
    const size = embedding.byteLength;
    this.embeddingCache.set(text, embedding, size);
  }

  /**
   * Get a cached embedding
   */
  getCachedEmbedding(text: string): Float32Array | undefined {
    return this.embeddingCache.get(text);
  }

  /**
   * Cache index data
   */
  cacheIndex(key: string, data: any): void {
    const size = this.estimateObjectSize(data);
    this.indexCache.set(key, data, size);
  }

  /**
   * Get cached index data
   */
  getCachedIndex(key: string): any | undefined {
    return this.indexCache.get(key);
  }

  /**
   * Mark index as loaded (for lazy loading)
   */
  markIndexLoaded(): void {
    this.indexLoaded = true;
  }

  /**
   * Check if index is loaded
   */
  isIndexLoaded(): boolean {
    return this.indexLoaded || !this.config.lazyLoadIndex;
  }

  /**
   * Mark models as loaded (for lazy loading)
   */
  markModelsLoaded(): void {
    this.modelsLoaded = true;
  }

  /**
   * Check if models are loaded
   */
  areModelsLoaded(): boolean {
    return this.modelsLoaded || !this.config.lazyLoadModels;
  }

  /**
   * Get performance statistics
   */
  getStats(): {
    memory: any;
    caches: {
      vectors: any;
      embeddings: any;
      index: any;
    };
    workers?: {
      available: number;
      pending: number;
    };
    batch?: {
      pending: number;
    };
  } {
    const stats: any = {
      memory: this.memoryManager.getMemoryStats(),
      caches: {
        vectors: this.vectorCache.getStats(),
        embeddings: this.embeddingCache.getStats(),
        index: this.indexCache.getStats(),
      },
    };

    if (this.workerPool) {
      stats.workers = {
        available: this.workerPool.getAvailableWorkerCount(),
        pending: this.workerPool.getPendingTaskCount(),
      };
    }

    if (this.batchOptimizer) {
      stats.batch = {
        pending: this.batchOptimizer.getPendingCount(),
      };
    }

    return stats;
  }

  /**
   * Clear all caches
   */
  clearCaches(): void {
    this.vectorCache.clear();
    this.embeddingCache.clear();
    this.indexCache.clear();
  }

  /**
   * Dispose of all resources
   */
  async dispose(): Promise<void> {
    this.memoryManager.stopMonitoring();
    this.clearCaches();

    if (this.workerPool) {
      this.workerPool.dispose();
    }

    if (this.batchOptimizer) {
      await this.batchOptimizer.flush();
      this.batchOptimizer.dispose();
    }

    this.initialized = false;
    this.indexLoaded = false;
    this.modelsLoaded = false;
  }

  /**
   * Estimate the size of a vector record in bytes
   */
  private estimateVectorSize(record: VectorRecord): number {
    // Vector size + metadata size estimate
    const vectorSize = record.vector.byteLength;
    const metadataSize = this.estimateObjectSize(record.metadata);
    return vectorSize + metadataSize + 100; // +100 for overhead
  }

  /**
   * Estimate the size of an object in bytes
   */
  private estimateObjectSize(obj: any): number {
    const str = JSON.stringify(obj);
    return str.length * 2; // Rough estimate (UTF-16)
  }
}
