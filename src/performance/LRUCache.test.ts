/**
 * Tests for LRUCache
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { LRUCache } from './LRUCache';

describe('LRUCache', () => {
  let cache: LRUCache<string>;

  beforeEach(() => {
    cache = new LRUCache<string>({
      maxSize: 1000,
      maxEntries: 5,
    });
  });

  it('should store and retrieve values', () => {
    cache.set('key1', 'value1', 100);
    expect(cache.get('key1')).toBe('value1');
  });

  it('should return undefined for non-existent keys', () => {
    expect(cache.get('nonexistent')).toBeUndefined();
  });

  it('should evict LRU entry when size limit is reached', () => {
    cache.set('key1', 'value1', 400);
    cache.set('key2', 'value2', 400);
    cache.set('key3', 'value3', 400); // Should evict key1

    expect(cache.get('key1')).toBeUndefined();
    expect(cache.get('key2')).toBe('value2');
    expect(cache.get('key3')).toBe('value3');
  });

  it('should evict LRU entry when entry limit is reached', () => {
    for (let i = 0; i < 6; i++) {
      cache.set(`key${i}`, `value${i}`, 50);
    }

    // key0 should be evicted
    expect(cache.get('key0')).toBeUndefined();
    expect(cache.get('key5')).toBe('value5');
  });

  it('should update access order on get', () => {
    cache.set('key1', 'value1', 100);
    cache.set('key2', 'value2', 100);
    cache.set('key3', 'value3', 100);

    // Access key1 to make it most recently used
    cache.get('key1');

    // Add more entries to trigger eviction
    cache.set('key4', 'value4', 400);
    cache.set('key5', 'value5', 400);

    // key1 should still be in cache (was accessed recently)
    // key2 and key3 should be evicted
    expect(cache.get('key1')).toBe('value1');
  });

  it('should update existing entries', () => {
    cache.set('key1', 'value1', 100);
    cache.set('key1', 'value2', 150);

    expect(cache.get('key1')).toBe('value2');
    expect(cache.size()).toBe(150);
  });

  it('should delete entries', () => {
    cache.set('key1', 'value1', 100);
    expect(cache.delete('key1')).toBe(true);
    expect(cache.get('key1')).toBeUndefined();
    expect(cache.size()).toBe(0);
  });

  it('should return false when deleting non-existent entry', () => {
    expect(cache.delete('nonexistent')).toBe(false);
  });

  it('should clear all entries', () => {
    cache.set('key1', 'value1', 100);
    cache.set('key2', 'value2', 100);
    cache.clear();

    expect(cache.count()).toBe(0);
    expect(cache.size()).toBe(0);
  });

  it('should call onEvict callback', () => {
    let evictedKey: string | null = null;
    const cacheWithCallback = new LRUCache<string>({
      maxSize: 500,
      onEvict: (key) => {
        evictedKey = key;
      },
    });

    cacheWithCallback.set('key1', 'value1', 300);
    cacheWithCallback.set('key2', 'value2', 300); // Should evict key1

    expect(evictedKey).toBe('key1');
  });

  it('should provide accurate statistics', () => {
    cache.set('key1', 'value1', 100);
    cache.set('key2', 'value2', 200);

    const stats = cache.getStats();
    expect(stats.size).toBe(300);
    expect(stats.count).toBe(2);
    expect(stats.maxSize).toBe(1000);
    expect(stats.utilizationPercent).toBe(30);
  });

  it('should check if key exists', () => {
    cache.set('key1', 'value1', 100);
    expect(cache.has('key1')).toBe(true);
    expect(cache.has('key2')).toBe(false);
  });
});
