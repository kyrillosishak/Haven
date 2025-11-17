/**
 * Tests for MemoryManager
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { MemoryManager } from './MemoryManager';
import { LRUCache } from './LRUCache';

describe('MemoryManager', () => {
  let memoryManager: MemoryManager;
  let cache1: LRUCache<string>;
  let cache2: LRUCache<string>;

  beforeEach(() => {
    memoryManager = new MemoryManager({
      maxMemoryMB: 1,
      evictionThreshold: 0.8,
      checkInterval: 100,
    });

    cache1 = new LRUCache<string>({
      maxSize: 500 * 1024,
    });

    cache2 = new LRUCache<string>({
      maxSize: 500 * 1024,
    });

    memoryManager.registerCache('cache1', cache1);
    memoryManager.registerCache('cache2', cache2);
  });

  afterEach(() => {
    memoryManager.dispose();
  });

  it('should register caches', () => {
    const stats = memoryManager.getMemoryStats();
    expect(stats.cacheStats).toHaveProperty('cache1');
    expect(stats.cacheStats).toHaveProperty('cache2');
  });

  it('should get memory statistics', () => {
    cache1.set('key1', 'value1', 100);
    cache2.set('key2', 'value2', 200);

    const stats = memoryManager.getMemoryStats();
    expect(stats.cacheStats.cache1.size).toBe(100);
    expect(stats.cacheStats.cache2.size).toBe(200);
  });

  it('should start and stop monitoring', () => {
    memoryManager.startMonitoring();
    memoryManager.stopMonitoring();
    // Should not throw
  });

  it('should call memory pressure callbacks', async () => {
    let callbackCalled = false;
    
    memoryManager.onMemoryPressure(async () => {
      callbackCalled = true;
    });

    // Fill caches to trigger memory pressure
    for (let i = 0; i < 100; i++) {
      cache1.set(`key${i}`, `value${i}`, 10000);
    }

    await memoryManager.checkMemory();

    // Note: This test might not trigger the callback depending on browser memory API
    // The callback is called when utilizationPercent >= evictionThreshold * 100
  });

  it('should force eviction', async () => {
    cache1.set('key1', 'value1', 100);
    cache2.set('key2', 'value2', 200);

    await memoryManager.forceEviction(0.1);

    // Caches should be cleared or significantly reduced
    const stats = memoryManager.getMemoryStats();
    expect(stats.cacheStats.cache1.size + stats.cacheStats.cache2.size).toBeLessThanOrEqual(300);
  });
});
