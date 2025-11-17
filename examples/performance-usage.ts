/**
 * Example: Using Performance Optimizations
 * 
 * This example demonstrates how to use the performance optimization features
 * including caching, batching, and progressive loading.
 */

import { VectorDB } from '../src/core/VectorDB';
import type { VectorDBConfig } from '../src/core/types';

async function main() {
  // Configure VectorDB with performance optimizations
  const config: VectorDBConfig = {
    storage: {
      dbName: 'performance-demo',
      version: 1,
    },
    index: {
      dimensions: 384,
      metric: 'cosine',
      indexType: 'kdtree',
    },
    embedding: {
      model: 'Xenova/all-MiniLM-L6-v2',
      device: 'wasm',
      cache: true,
    },
    // Performance configuration
    performance: {
      maxMemoryMB: 500,
      evictionThreshold: 0.9,
      vectorCacheSize: 100 * 1024 * 1024, // 100MB
      embeddingCacheSize: 50 * 1024 * 1024, // 50MB
      indexCacheSize: 100 * 1024 * 1024, // 100MB
      enableWorkers: false, // Disable for this example
      batchSize: 100,
      batchWaitTime: 100,
      chunkSize: 1000,
      lazyLoadIndex: false,
      lazyLoadModels: false,
    },
  };

  console.log('Initializing VectorDB with performance optimizations...');
  const db = new VectorDB(config);
  await db.initialize();

  // Example 1: Batch Insert with Caching
  console.log('\n1. Batch Insert with Automatic Caching');
  const documents = [
    { text: 'Machine learning is fascinating', metadata: { category: 'AI' } },
    { text: 'Deep learning powers modern AI', metadata: { category: 'AI' } },
    { text: 'Natural language processing is evolving', metadata: { category: 'NLP' } },
    { text: 'Computer vision enables image recognition', metadata: { category: 'CV' } },
    { text: 'Reinforcement learning trains agents', metadata: { category: 'RL' } },
  ];

  const ids = await db.insertBatch(documents);
  console.log(`Inserted ${ids.length} documents`);

  // Example 2: Search with Embedding Cache
  console.log('\n2. Search with Embedding Cache');
  const query = 'artificial intelligence';
  
  // First search - generates embedding
  console.time('First search (no cache)');
  const results1 = await db.search({ text: query, k: 3 });
  console.timeEnd('First search (no cache)');
  console.log(`Found ${results1.length} results`);

  // Second search - uses cached embedding
  console.time('Second search (cached)');
  const results2 = await db.search({ text: query, k: 3 });
  console.timeEnd('Second search (cached)');
  console.log(`Found ${results2.length} results (from cache)`);

  // Example 3: Performance Statistics
  console.log('\n3. Performance Statistics');
  const stats = db.getPerformanceStats();
  console.log('Memory Stats:', {
    usedMemory: `${(stats.memory.usedMemory / 1024 / 1024).toFixed(2)} MB`,
    utilization: `${stats.memory.utilizationPercent.toFixed(2)}%`,
  });
  console.log('Cache Stats:', {
    vectors: {
      count: stats.caches.vectors.count,
      size: `${(stats.caches.vectors.size / 1024 / 1024).toFixed(2)} MB`,
      utilization: `${stats.caches.vectors.utilizationPercent.toFixed(2)}%`,
    },
    embeddings: {
      count: stats.caches.embeddings.count,
      size: `${(stats.caches.embeddings.size / 1024 / 1024).toFixed(2)} MB`,
      utilization: `${stats.caches.embeddings.utilizationPercent.toFixed(2)}%`,
    },
  });

  // Example 4: Export with Progressive Loading
  console.log('\n4. Export with Progressive Loading');
  console.time('Export');
  const exportData = await db.export();
  console.timeEnd('Export');
  console.log(`Exported ${exportData.vectors.length} vectors`);

  // Example 5: Clear Caches
  console.log('\n5. Clear Caches');
  db.clearCaches();
  console.log('Caches cleared');

  // Verify cache is cleared
  console.time('Search after cache clear');
  await db.search({ text: query, k: 3 });
  console.timeEnd('Search after cache clear');

  // Clean up
  await db.dispose();
  console.log('\nDemo complete!');
}

// Run the example
main().catch(console.error);
