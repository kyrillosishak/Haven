/**
 * LRU (Least Recently Used) Cache implementation
 * Used for caching vectors, embeddings, and index data
 */

export interface CacheEntry<T> {
  value: T;
  size: number;
  timestamp: number;
}

export interface LRUCacheConfig {
  maxSize: number; // Maximum size in bytes
  maxEntries?: number; // Maximum number of entries
  onEvict?: (key: string, value: any) => void;
}

/**
 * Generic LRU Cache with size-based eviction
 */
export class LRUCache<T> {
  private cache: Map<string, CacheEntry<T>>;
  private accessOrder: string[]; // Track access order for LRU
  private currentSize: number = 0;
  private config: Required<LRUCacheConfig>;

  constructor(config: LRUCacheConfig) {
    this.cache = new Map();
    this.accessOrder = [];
    this.config = {
      maxEntries: Infinity,
      onEvict: () => {},
      ...config,
    };
  }

  /**
   * Get a value from the cache
   */
  get(key: string): T | undefined {
    const entry = this.cache.get(key);
    
    if (!entry) {
      return undefined;
    }

    // Update access order (move to end = most recently used)
    this.updateAccessOrder(key);
    entry.timestamp = Date.now();

    return entry.value;
  }

  /**
   * Set a value in the cache
   */
  set(key: string, value: T, size: number): void {
    // Check if key already exists
    const existing = this.cache.get(key);
    if (existing) {
      // Update existing entry
      this.currentSize -= existing.size;
      this.currentSize += size;
      existing.value = value;
      existing.size = size;
      existing.timestamp = Date.now();
      this.updateAccessOrder(key);
      return;
    }

    // Evict entries if necessary
    while (
      (this.currentSize + size > this.config.maxSize ||
        this.cache.size >= this.config.maxEntries) &&
      this.cache.size > 0
    ) {
      this.evictLRU();
    }

    // Add new entry
    this.cache.set(key, {
      value,
      size,
      timestamp: Date.now(),
    });
    this.accessOrder.push(key);
    this.currentSize += size;
  }

  /**
   * Check if a key exists in the cache
   */
  has(key: string): boolean {
    return this.cache.has(key);
  }

  /**
   * Delete a specific entry
   */
  delete(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) {
      return false;
    }

    this.cache.delete(key);
    this.currentSize -= entry.size;
    this.accessOrder = this.accessOrder.filter(k => k !== key);
    this.config.onEvict(key, entry.value);

    return true;
  }

  /**
   * Clear all entries
   */
  clear(): void {
    for (const [key, entry] of this.cache.entries()) {
      this.config.onEvict(key, entry.value);
    }
    this.cache.clear();
    this.accessOrder = [];
    this.currentSize = 0;
  }

  /**
   * Get current cache size in bytes
   */
  size(): number {
    return this.currentSize;
  }

  /**
   * Get number of entries
   */
  count(): number {
    return this.cache.size;
  }

  /**
   * Get cache statistics
   */
  getStats(): {
    size: number;
    count: number;
    maxSize: number;
    maxEntries: number;
    utilizationPercent: number;
  } {
    return {
      size: this.currentSize,
      count: this.cache.size,
      maxSize: this.config.maxSize,
      maxEntries: this.config.maxEntries,
      utilizationPercent: (this.currentSize / this.config.maxSize) * 100,
    };
  }

  /**
   * Evict the least recently used entry
   */
  private evictLRU(): void {
    if (this.accessOrder.length === 0) {
      return;
    }

    const lruKey = this.accessOrder.shift()!;
    const entry = this.cache.get(lruKey);

    if (entry) {
      this.cache.delete(lruKey);
      this.currentSize -= entry.size;
      this.config.onEvict(lruKey, entry.value);
    }
  }

  /**
   * Update access order for a key (move to end)
   */
  private updateAccessOrder(key: string): void {
    const index = this.accessOrder.indexOf(key);
    if (index > -1) {
      this.accessOrder.splice(index, 1);
    }
    this.accessOrder.push(key);
  }
}
