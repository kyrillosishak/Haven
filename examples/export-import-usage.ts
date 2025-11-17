/**
 * Example: Data Export and Import
 * 
 * Demonstrates how to export and import vector database data
 * with progress tracking and validation.
 */

import { VectorDB } from '../src/index';
import type { ExportData } from '../src/core/types';

async function main() {
  console.log('=== VectorDB Export/Import Example ===\n');

  // Initialize database
  const db = new VectorDB({
    storage: { dbName: 'export-import-demo' },
    index: { dimensions: 384, metric: 'cosine', indexType: 'kdtree' },
    embedding: {
      model: 'Xenova/all-MiniLM-L6-v2',
      device: 'wasm',
      cache: true,
    },
    performance: {
      chunkSize: 100,
      lazyLoadModels: false,
    },
  });

  await db.initialize();
  console.log('✓ Database initialized\n');

  // Clear any existing data
  await db.clear();

  // Insert sample documents
  console.log('Inserting sample documents...');
  const documents = [
    { text: 'Machine learning is a subset of artificial intelligence', metadata: { category: 'AI', topic: 'ML' } },
    { text: 'Neural networks are inspired by biological neurons', metadata: { category: 'AI', topic: 'Neural Networks' } },
    { text: 'Deep learning uses multiple layers of neural networks', metadata: { category: 'AI', topic: 'Deep Learning' } },
    { text: 'Natural language processing enables computers to understand text', metadata: { category: 'NLP', topic: 'Text Processing' } },
    { text: 'Computer vision allows machines to interpret visual information', metadata: { category: 'CV', topic: 'Image Processing' } },
  ];

  await db.insertBatch(documents);
  console.log(`✓ Inserted ${documents.length} documents\n`);

  // Export database with progress tracking
  console.log('Exporting database...');
  let exportData: ExportData;
  
  exportData = await db.export({
    includeIndex: true,
    onProgress: (loaded, total) => {
      const percent = Math.round((loaded / total) * 100);
      console.log(`  Export progress: ${loaded}/${total} (${percent}%)`);
    },
  });

  console.log('\n✓ Export completed');
  console.log(`  Version: ${exportData.version}`);
  console.log(`  Vector count: ${exportData.metadata.vectorCount}`);
  console.log(`  Dimensions: ${exportData.metadata.dimensions}`);
  console.log(`  Exported at: ${new Date(exportData.metadata.exportedAt).toISOString()}`);
  console.log(`  Export size: ${JSON.stringify(exportData).length} bytes\n`);

  // Save export to file (in Node.js environment)
  if (typeof process !== 'undefined' && process.versions?.node) {
    const fs = await import('fs');
    const exportPath = './vectordb-export.json';
    fs.writeFileSync(exportPath, JSON.stringify(exportData, null, 2));
    console.log(`✓ Export saved to ${exportPath}\n`);
  }

  // Clear database
  console.log('Clearing database...');
  await db.clear();
  console.log(`✓ Database cleared (size: ${await db.size()})\n`);

  // Import database with progress tracking
  console.log('Importing database...');
  await db.import(exportData, {
    validateSchema: true,
    clearExisting: true,
    onProgress: (loaded, total) => {
      const percent = Math.round((loaded / total) * 100);
      console.log(`  Import progress: ${loaded}/${total} (${percent}%)`);
    },
  });

  console.log('\n✓ Import completed');
  console.log(`  Database size: ${await db.size()}\n`);

  // Verify imported data with search
  console.log('Verifying imported data...');
  const results = await db.search({
    text: 'artificial intelligence and neural networks',
    k: 3,
  });

  console.log(`Found ${results.length} results:`);
  results.forEach((result, i) => {
    console.log(`\n${i + 1}. Score: ${result.score.toFixed(4)}`);
    console.log(`   Category: ${result.metadata.category}`);
    console.log(`   Topic: ${result.metadata.topic}`);
    console.log(`   Content: ${result.metadata.content?.substring(0, 60)}...`);
  });

  // Example: Export without index (smaller file size)
  console.log('\n\nExporting without index (for smaller file size)...');
  const lightExport = await db.export({
    includeIndex: false,
  });

  const fullSize = JSON.stringify(exportData).length;
  const lightSize = JSON.stringify(lightExport).length;
  const savings = Math.round(((fullSize - lightSize) / fullSize) * 100);

  console.log(`✓ Light export completed`);
  console.log(`  Full export size: ${fullSize} bytes`);
  console.log(`  Light export size: ${lightSize} bytes`);
  console.log(`  Size reduction: ${savings}%\n`);

  // Example: Partial import (without clearing existing data)
  console.log('Demonstrating partial import...');
  const partialData: ExportData = {
    version: '1.0.0',
    config: exportData.config,
    vectors: [
      {
        id: 'new-doc-1',
        vector: Array(384).fill(0.1),
        metadata: { category: 'Robotics', topic: 'Automation', content: 'Robots can perform tasks autonomously' },
        timestamp: Date.now(),
      },
    ],
    index: '',
    metadata: {
      exportedAt: Date.now(),
      vectorCount: 1,
      dimensions: 384,
    },
  };

  const sizeBefore = await db.size();
  await db.import(partialData, {
    clearExisting: false, // Keep existing data
    validateSchema: true,
  });
  const sizeAfter = await db.size();

  console.log(`✓ Partial import completed`);
  console.log(`  Size before: ${sizeBefore}`);
  console.log(`  Size after: ${sizeAfter}`);
  console.log(`  Added: ${sizeAfter - sizeBefore} documents\n`);

  // Example: Handling version compatibility
  console.log('Testing version compatibility...');
  try {
    const incompatibleData: ExportData = {
      ...exportData,
      version: '2.0.0', // Future version
    };
    await db.import(incompatibleData);
  } catch (error: any) {
    console.log(`✓ Version check working: ${error.message}\n`);
  }

  // Example: Handling dimension mismatch
  console.log('Testing dimension validation...');
  try {
    const wrongDimensionData: ExportData = {
      ...exportData,
      metadata: {
        ...exportData.metadata,
        dimensions: 512, // Wrong dimensions
      },
    };
    await db.import(wrongDimensionData);
  } catch (error: any) {
    console.log(`✓ Dimension validation working: ${error.message}\n`);
  }

  // Clean up
  await db.dispose();
  console.log('✓ Database disposed\n');

  console.log('=== Example completed successfully ===');
}

// Run example
main().catch(console.error);
