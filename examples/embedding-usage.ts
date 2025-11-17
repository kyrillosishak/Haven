/**
 * Example: Using TransformersEmbedding for text embeddings
 */

import { TransformersEmbedding } from '../src/embedding/TransformersEmbedding';

async function main() {
  console.log('=== Transformers.js Embedding Example ===\n');

  // Create embedding generator with configuration
  const embedding = new TransformersEmbedding({
    model: 'Xenova/all-MiniLM-L6-v2', // 384-dimensional embeddings
    device: 'wasm', // Use WASM backend (or 'webgpu' for GPU acceleration)
    cache: true, // Enable model caching
    quantized: true, // Use quantized model for smaller size
  });

  try {
    // Initialize the model (downloads and loads the model)
    console.log('Initializing embedding model...');
    await embedding.initialize();
    console.log(`Model loaded! Embedding dimensions: ${embedding.getDimensions()}\n`);

    // Generate embedding for a single text
    console.log('--- Single Text Embedding ---');
    const text1 = 'Machine learning is a subset of artificial intelligence';
    const vector1 = await embedding.embed(text1);
    console.log(`Text: "${text1}"`);
    console.log(`Embedding: [${vector1.slice(0, 5).join(', ')}...] (${vector1.length} dimensions)\n`);

    // Generate embeddings for multiple texts in batch
    console.log('--- Batch Text Embedding ---');
    const texts = [
      'Deep learning uses neural networks',
      'Natural language processing analyzes text',
      'Computer vision processes images',
    ];
    const vectors = await embedding.embedBatch(texts);
    console.log(`Generated ${vectors.length} embeddings:`);
    texts.forEach((text, i) => {
      console.log(`  ${i + 1}. "${text}"`);
      console.log(`     Vector: [${vectors[i].slice(0, 3).join(', ')}...]\n`);
    });

    // Calculate similarity between embeddings
    console.log('--- Similarity Calculation ---');
    const text2 = 'Neural networks are used in deep learning';
    const vector2 = await embedding.embed(text2);
    
    const similarity = cosineSimilarity(vector1, vector2);
    console.log(`Text 1: "${text1}"`);
    console.log(`Text 2: "${text2}"`);
    console.log(`Cosine Similarity: ${similarity.toFixed(4)}\n`);

    // Compare with unrelated text
    const text3 = 'Cooking pasta requires boiling water';
    const vector3 = await embedding.embed(text3);
    const similarity2 = cosineSimilarity(vector1, vector3);
    console.log(`Text 1: "${text1}"`);
    console.log(`Text 3: "${text3}"`);
    console.log(`Cosine Similarity: ${similarity2.toFixed(4)}\n`);

    // Clean up resources
    await embedding.dispose();
    console.log('Resources cleaned up successfully!');

  } catch (error) {
    console.error('Error:', error);
  }
}

/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Run the example
main().catch(console.error);
