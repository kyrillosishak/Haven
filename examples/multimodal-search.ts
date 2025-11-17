/**
 * Multimodal Search Example
 * 
 * This example demonstrates how to use the VectorDB for multimodal search,
 * combining text and image embeddings using CLIP models.
 */

import { VectorDB } from '../src/index';

async function main() {
  console.log('=== Multimodal Search Example ===\n');

  // Step 1: Initialize VectorDB with CLIP model for multimodal embeddings
  console.log('1. Initializing VectorDB with CLIP model...');
  const vectorDB = new VectorDB({
    storage: {
      dbName: 'multimodal-search-db',
      version: 1,
    },
    index: {
      dimensions: 512, // CLIP embeddings are 512-dimensional
      metric: 'cosine',
      indexType: 'kdtree',
    },
    embedding: {
      model: 'Xenova/clip-vit-base-patch32', // CLIP model for text and images
      device: 'wasm',
      cache: true,
    },
  });

  await vectorDB.initialize();
  console.log('✓ VectorDB initialized with CLIP model\n');

  // Step 2: Insert text descriptions
  console.log('2. Inserting text descriptions...');
  const textDocuments = [
    {
      text: 'A beautiful sunset over the ocean with orange and pink colors',
      metadata: {
        type: 'text',
        category: 'nature',
        title: 'Ocean Sunset',
        tags: ['sunset', 'ocean', 'nature'],
      },
    },
    {
      text: 'A cute golden retriever puppy playing in the park',
      metadata: {
        type: 'text',
        category: 'animals',
        title: 'Puppy in Park',
        tags: ['dog', 'puppy', 'park'],
      },
    },
    {
      text: 'Modern city skyline at night with illuminated skyscrapers',
      metadata: {
        type: 'text',
        category: 'urban',
        title: 'City Skyline',
        tags: ['city', 'night', 'architecture'],
      },
    },
    {
      text: 'Fresh vegetables and fruits on a wooden table',
      metadata: {
        type: 'text',
        category: 'food',
        title: 'Fresh Produce',
        tags: ['food', 'healthy', 'vegetables'],
      },
    },
    {
      text: 'Mountain landscape with snow-capped peaks and blue sky',
      metadata: {
        type: 'text',
        category: 'nature',
        title: 'Mountain Vista',
        tags: ['mountains', 'landscape', 'nature'],
      },
    },
  ];

  const textIds = await vectorDB.insertBatch(textDocuments);
  console.log(`✓ Inserted ${textIds.length} text descriptions\n`);

  // Step 3: Demonstrate text-to-text search
  console.log('3. Text-to-Text Search...');
  console.log('Query: "beautiful landscape with mountains"\n');

  const textResults = await vectorDB.search({
    text: 'beautiful landscape with mountains',
    k: 3,
  });

  console.log('Top Results:');
  textResults.forEach((result, index) => {
    console.log(`${index + 1}. ${result.metadata.title} (Score: ${result.score.toFixed(4)})`);
    console.log(`   ${result.metadata.text}`);
    console.log(`   Category: ${result.metadata.category}`);
    console.log();
  });

  // Step 4: Demonstrate cross-modal search (text query for image content)
  console.log('4. Cross-Modal Search (Text → Images)...');
  console.log('Note: In a real implementation, you would have actual images stored.\n');
  console.log('Query: "dog playing outdoors"\n');

  const crossModalResults = await vectorDB.search({
    text: 'dog playing outdoors',
    k: 3,
    filter: {
      field: 'category',
      operator: 'eq',
      value: 'animals',
    },
  });

  console.log('Matching Images:');
  crossModalResults.forEach((result, index) => {
    console.log(`${index + 1}. ${result.metadata.title} (Score: ${result.score.toFixed(4)})`);
    console.log(`   Description: ${result.metadata.text}`);
    console.log();
  });

  // Step 5: Demonstrate image embedding (conceptual)
  console.log('5. Image Embedding (Conceptual)...\n');
  console.log('To embed an image, you would use:');
  console.log(`
// Load image from file or canvas
const imageBlob = await fetch('path/to/image.jpg').then(r => r.blob());

// Generate embedding
const embeddingGenerator = vectorDB.getEmbeddingGenerator();
const imageEmbedding = await embeddingGenerator.embedImage(imageBlob);

// Insert with embedding
await vectorDB.insert({
  vector: imageEmbedding,
  metadata: {
    type: 'image',
    url: 'path/to/image.jpg',
    caption: 'Image description',
    category: 'nature',
  },
});
  `);

  // Step 6: Demonstrate filtering by modality
  console.log('\n6. Filtering by Modality...');
  
  const textOnlyResults = await vectorDB.search({
    text: 'nature scenery',
    k: 5,
    filter: {
      field: 'type',
      operator: 'eq',
      value: 'text',
    },
  });

  console.log(`Found ${textOnlyResults.length} text-only results for "nature scenery"`);
  textOnlyResults.forEach((result, index) => {
    console.log(`  ${index + 1}. ${result.metadata.title}`);
  });
  console.log();

  // Step 7: Demonstrate semantic similarity across categories
  console.log('7. Semantic Similarity Across Categories...');
  console.log('Query: "outdoor scenery"\n');

  const semanticResults = await vectorDB.search({
    text: 'outdoor scenery',
    k: 5,
  });

  console.log('Results from different categories:');
  semanticResults.forEach((result, index) => {
    console.log(`${index + 1}. [${result.metadata.category}] ${result.metadata.title}`);
    console.log(`   Score: ${result.score.toFixed(4)}`);
  });
  console.log();

  // Step 8: Batch search for multiple queries
  console.log('8. Batch Search for Multiple Queries...\n');

  const queries = [
    'animals in nature',
    'urban architecture',
    'healthy food',
  ];

  console.log('Searching for multiple queries:');
  for (const query of queries) {
    const results = await vectorDB.search({
      text: query,
      k: 1,
    });

    if (results.length > 0) {
      console.log(`  "${query}" → ${results[0].metadata.title} (${results[0].score.toFixed(4)})`);
    }
  }
  console.log();

  // Step 9: Advanced filtering with compound conditions
  console.log('9. Advanced Filtering...');
  console.log('Query: "nature" with category filter\n');

  const filteredResults = await vectorDB.search({
    text: 'nature',
    k: 10,
    filter: {
      field: 'category',
      operator: 'in',
      value: ['nature', 'animals'],
    },
  });

  console.log(`Found ${filteredResults.length} results in nature/animals categories:`);
  filteredResults.forEach((result, index) => {
    console.log(`  ${index + 1}. ${result.metadata.title} (${result.metadata.category})`);
  });
  console.log();

  // Step 10: Statistics and insights
  console.log('10. Database Statistics...');
  const size = await vectorDB.size();
  console.log(`  Total documents: ${size}`);
  console.log(`  Embedding dimensions: 512 (CLIP)`);
  console.log(`  Similarity metric: cosine`);
  console.log();

  // Step 11: Use cases
  console.log('11. Multimodal Search Use Cases:\n');
  console.log('  📸 Image Search: Find images using text descriptions');
  console.log('  🔍 Reverse Image Search: Find similar images');
  console.log('  📝 Caption Generation: Find relevant text for images');
  console.log('  🎨 Content Discovery: Explore related content across modalities');
  console.log('  🏷️  Auto-Tagging: Automatically tag images based on content');
  console.log('  🔗 Cross-Modal Retrieval: Bridge text and visual content');
  console.log();

  // Step 12: Best practices
  console.log('12. Best Practices:\n');
  console.log('  ✓ Use CLIP models for text-image compatibility');
  console.log('  ✓ Normalize embeddings for better similarity scores');
  console.log('  ✓ Store both text descriptions and image URLs');
  console.log('  ✓ Use metadata filtering to separate modalities');
  console.log('  ✓ Consider image preprocessing (resize, normalize)');
  console.log('  ✓ Cache embeddings to avoid recomputation');
  console.log();

  // Cleanup
  console.log('13. Cleanup...');
  await vectorDB.dispose();
  console.log('✓ Resources cleaned up\n');

  console.log('=== Multimodal Search Example Complete ===');
  console.log('\nNext Steps:');
  console.log('1. Add actual image files to your database');
  console.log('2. Implement image preprocessing pipeline');
  console.log('3. Build a visual search interface');
  console.log('4. Experiment with different CLIP model variants');
}

// Run the example
main().catch(console.error);
