/**
 * IndexManager - Manages vector indexing using Voy WASM engine
 */

import { Voy, type Resource, type EmbeddedResource } from 'voy-search/voy_search';
import type { VectorRecord, MetadataFilter, CompoundFilter, Filter, StorageManager } from '../storage/types';
import type { SearchResult, IndexStats } from './types';
import { VectorDBError, DimensionMismatchError, IndexCorruptedError, InputValidator } from '../errors';

export interface IndexManagerConfig {
  dimensions: number;
  metric: 'cosine' | 'euclidean' | 'dot';
  storage: StorageManager;
}

export class IndexManager {
  private index: Voy | null = null;
  private config: IndexManagerConfig;
  private vectorCount: number = 0;
  private lastUpdated: number = 0;
  private isInitialized: boolean = false;

  constructor(config: IndexManagerConfig) {
    this.config = config;
  }

  /**
   * Initialize the index manager, optionally loading from storage
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    try {
      // Try to load existing index from storage
      const serializedIndex = await this.config.storage.loadIndex();
      
      if (serializedIndex) {
        await this.deserialize(serializedIndex);
      } else {
        // Create new empty index
        this.index = new Voy();
      }

      this.isInitialized = true;
    } catch (error) {
      if (error instanceof IndexCorruptedError) {
        // If index is corrupted, rebuild from vectors
        console.warn('Index corrupted, will rebuild on first search');
        this.index = new Voy();
        this.isInitialized = true;
      } else {
        throw new VectorDBError(
          'Failed to initialize index',
          'INDEX_INIT_ERROR',
          { error }
        );
      }
    }
  }

  /**
   * Build index from a collection of vectors
   */
  async build(vectors: VectorRecord[]): Promise<void> {
    this.ensureInitialized();

    if (vectors.length === 0) {
      return;
    }

    // Validate dimensions
    for (const vector of vectors) {
      InputValidator.validateVector(vector.vector, this.config.dimensions);
    }

    try {
      // Convert vectors to Voy resource format
      const embeddedResources: EmbeddedResource[] = vectors.map(v => ({
        id: v.id,
        title: v.metadata.title || v.metadata.content || v.id,
        url: v.metadata.url || '',
        embeddings: Array.from(v.vector)
      }));

      const resource: Resource = {
        embeddings: embeddedResources
      };

      // Create new index with initial data
      this.index = new Voy(resource);

      this.vectorCount = vectors.length;
      this.lastUpdated = Date.now();

      // Persist index to storage
      await this.persistIndex();
    } catch (error) {
      if (error instanceof VectorDBError) {
        throw error;
      }
      throw new VectorDBError(
        'Failed to build index',
        'INDEX_BUILD_ERROR',
        { error, vectorCount: vectors.length }
      );
    }
  }

  /**
   * Add a single vector to the index
   */
  async add(vector: VectorRecord): Promise<void> {
    this.ensureInitialized();

    InputValidator.validateVector(vector.vector, this.config.dimensions);

    try {
      const embeddedResource: EmbeddedResource = {
        id: vector.id,
        title: vector.metadata.title || vector.metadata.content || vector.id,
        url: vector.metadata.url || '',
        embeddings: Array.from(vector.vector)
      };

      const resource: Resource = {
        embeddings: [embeddedResource]
      };

      // If this is the first vector, create a new index with it
      // Voy requires at least one vector to establish dimensions
      if (this.vectorCount === 0) {
        this.index = new Voy(resource);
      } else {
        // Add to existing Voy index
        this.index!.add(resource);
      }

      this.vectorCount++;
      this.lastUpdated = Date.now();

      // Persist index to storage
      await this.persistIndex();
    } catch (error) {
      if (error instanceof VectorDBError) {
        throw error;
      }
      throw new VectorDBError(
        'Failed to add vector to index',
        'INDEX_ADD_ERROR',
        { error, vectorId: vector.id }
      );
    }
  }

  /**
   * Add multiple vectors to the index in batch
   */
  async addBatch(vectors: VectorRecord[]): Promise<void> {
    this.ensureInitialized();

    if (vectors.length === 0) {
      return;
    }

    // Validate dimensions
    for (const vector of vectors) {
      InputValidator.validateVector(vector.vector, this.config.dimensions);
    }

    try {
      const embeddedResources: EmbeddedResource[] = vectors.map(v => ({
        id: v.id,
        title: v.metadata.title || v.metadata.content || v.id,
        url: v.metadata.url || '',
        embeddings: Array.from(v.vector)
      }));

      const resource: Resource = {
        embeddings: embeddedResources
      };

      // If this is the first batch, create a new index with it
      // Voy requires at least one vector to establish dimensions
      if (this.vectorCount === 0) {
        this.index = new Voy(resource);
      } else {
        // Add all vectors to existing Voy index
        this.index!.add(resource);
      }

      this.vectorCount += vectors.length;
      this.lastUpdated = Date.now();

      // Persist index to storage
      await this.persistIndex();
    } catch (error) {
      if (error instanceof VectorDBError) {
        throw error;
      }
      throw new VectorDBError(
        'Failed to add vector batch to index',
        'INDEX_ADD_BATCH_ERROR',
        { error, vectorCount: vectors.length }
      );
    }
  }

  /**
   * Remove a vector from the index by ID
   * Note: Voy requires rebuilding the index to remove vectors
   */
  async remove(id: string): Promise<void> {
    this.ensureInitialized();

    try {
      // Fetch all vectors from storage
      const allVectors = await this.config.storage.getAll();

      // Filter out the vector to remove
      const remainingVectors = allVectors.filter(v => v.id !== id);

      // Rebuild the index without the removed vector
      if (remainingVectors.length > 0) {
        await this.build(remainingVectors);
      } else {
        // If no vectors remain, clear the index
        this.index!.clear();
        this.vectorCount = 0;
        this.lastUpdated = Date.now();
        await this.persistIndex();
      }
    } catch (error) {
      throw new VectorDBError(
        'Failed to remove vector from index',
        'INDEX_REMOVE_ERROR',
        { error, vectorId: id }
      );
    }
  }

  /**
   * Search for k nearest neighbors
   */
  async search(
    query: Float32Array,
    k: number,
    filter?: Filter
  ): Promise<SearchResult[]> {
    this.ensureInitialized();

    InputValidator.validateVector(query, this.config.dimensions);
    InputValidator.validateSearchQuery(k);

    if (this.vectorCount === 0) {
      return [];
    }

    try {
      // Perform search using Voy
      // Get more results to account for filtering
      const searchMultiplier = filter ? 3 : 1;
      const results = this.index!.search(query, k * searchMultiplier);

      // Map results to our format
      const searchResults: SearchResult[] = [];

      for (const neighbor of results.neighbors) {
        // Retrieve full record from storage
        const record = await this.config.storage.get(neighbor.id);
        
        if (!record) {
          continue;
        }

        // Apply metadata filter if provided
        if (filter && !this.evaluateFilter(record, filter)) {
          continue;
        }

        // Voy returns cosine similarity (higher is better, range 0-1)
        // We can use it directly as the score
        searchResults.push({
          id: record.id,
          score: 1.0, // Voy doesn't expose distance/score in the result
          metadata: record.metadata,
        });

        // Stop if we have enough results after filtering
        if (searchResults.length >= k) {
          break;
        }
      }

      return searchResults;
    } catch (error) {
      if (error instanceof VectorDBError) {
        throw error;
      }
      throw new VectorDBError(
        'Failed to search index',
        'INDEX_SEARCH_ERROR',
        { error, k }
      );
    }
  }

  /**
   * Serialize the index to a string
   */
  async serialize(): Promise<string> {
    this.ensureInitialized();

    try {
      // Use Voy's built-in serialization
      const voyIndex = this.index!.serialize();
      
      const indexData = {
        version: '1.0',
        dimensions: this.config.dimensions,
        metric: this.config.metric,
        vectorCount: this.vectorCount,
        lastUpdated: this.lastUpdated,
        voyIndex
      };

      return JSON.stringify(indexData);
    } catch (error) {
      throw new VectorDBError(
        'Failed to serialize index',
        'INDEX_SERIALIZE_ERROR',
        { error }
      );
    }
  }

  /**
   * Deserialize the index from a string
   */
  async deserialize(data: string): Promise<void> {
    try {
      const indexData = JSON.parse(data);

      // Validate format
      if (!indexData.version || !indexData.dimensions || !indexData.voyIndex) {
        throw new IndexCorruptedError({ reason: 'Invalid index format' });
      }

      // Validate dimensions match
      if (indexData.dimensions !== this.config.dimensions) {
        throw new DimensionMismatchError(this.config.dimensions, indexData.dimensions);
      }

      // Restore metadata
      this.vectorCount = indexData.vectorCount;
      this.lastUpdated = indexData.lastUpdated;

      // Deserialize Voy index
      this.index = Voy.deserialize(indexData.voyIndex);

    } catch (error) {
      if (error instanceof IndexCorruptedError || error instanceof DimensionMismatchError) {
        throw error;
      }
      throw new IndexCorruptedError({ error, reason: 'Failed to parse index data' });
    }
  }

  /**
   * Get index statistics
   */
  getStats(): IndexStats {
    return {
      vectorCount: this.vectorCount,
      dimensions: this.config.dimensions,
      indexType: 'voy',
      memoryUsage: this.estimateMemoryUsage(),
      lastUpdated: this.lastUpdated,
    };
  }

  /**
   * Clear the index
   */
  async clear(): Promise<void> {
    this.ensureInitialized();
    
    this.index!.clear();
    this.vectorCount = 0;
    this.lastUpdated = Date.now();

    await this.persistIndex();
  }

  /**
   * Persist index to storage
   */
  private async persistIndex(): Promise<void> {
    try {
      const serialized = await this.serialize();
      await this.config.storage.saveIndex(serialized);
    } catch (error) {
      // Log error but don't fail the operation
      console.error('Failed to persist index:', error);
    }
  }

  /**
   * Evaluate a filter (simple or compound) against a record
   */
  private evaluateFilter(record: VectorRecord, filter: Filter): boolean {
    if (this.isCompoundFilter(filter)) {
      return this.evaluateCompoundFilter(record, filter);
    } else {
      return this.matchesFilter(record, filter);
    }
  }

  /**
   * Type guard to check if a filter is a compound filter
   */
  private isCompoundFilter(filter: Filter): filter is CompoundFilter {
    return 'operator' in filter && ('and' === filter.operator || 'or' === filter.operator);
  }

  /**
   * Evaluate a compound filter (AND/OR logic)
   */
  private evaluateCompoundFilter(record: VectorRecord, filter: CompoundFilter): boolean {
    if (!filter.filters || filter.filters.length === 0) {
      return true;
    }

    if (filter.operator === 'and') {
      // All filters must match
      return filter.filters.every(f => this.evaluateFilter(record, f));
    } else if (filter.operator === 'or') {
      // At least one filter must match
      return filter.filters.some(f => this.evaluateFilter(record, f));
    }

    return false;
  }

  /**
   * Check if a record matches a metadata filter
   */
  private matchesFilter(record: VectorRecord, filter: MetadataFilter): boolean {
    const value = this.getNestedValue(record.metadata, filter.field);

    if (value === undefined) {
      return false;
    }

    switch (filter.operator) {
      case 'eq':
        return value === filter.value;
      case 'ne':
        return value !== filter.value;
      case 'gt':
        return value > filter.value;
      case 'gte':
        return value >= filter.value;
      case 'lt':
        return value < filter.value;
      case 'lte':
        return value <= filter.value;
      case 'in':
        return Array.isArray(filter.value) && filter.value.includes(value);
      case 'contains':
        if (Array.isArray(value)) {
          return value.includes(filter.value);
        }
        if (typeof value === 'string') {
          return value.includes(filter.value);
        }
        return false;
      default:
        return false;
    }
  }

  /**
   * Get nested value from object using dot notation
   */
  private getNestedValue(obj: any, path: string): any {
    const parts = path.split('.');
    let current = obj;

    for (const part of parts) {
      if (current === null || current === undefined) {
        return undefined;
      }
      current = current[part];
    }

    return current;
  }

  /**
   * Estimate memory usage in bytes
   */
  private estimateMemoryUsage(): number {
    // Rough estimate: 
    // - Each vector: dimensions * 4 bytes (Float32)
    // - Mappings: vectorCount * 100 bytes (rough estimate for Map overhead)
    const vectorMemory = this.vectorCount * this.config.dimensions * 4;
    const mappingMemory = this.vectorCount * 100;
    return vectorMemory + mappingMemory;
  }

  /**
   * Ensure the index is initialized
   */
  private ensureInitialized(): void {
    if (!this.isInitialized || !this.index) {
      throw new VectorDBError(
        'Index not initialized. Call initialize() first.',
        'INDEX_NOT_INITIALIZED'
      );
    }
  }
}
