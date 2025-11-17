/**
 * Example: Using WllamaProvider for local LLM inference
 */

import { WllamaProvider } from '../src/llm/WllamaProvider';

async function basicGeneration() {
  console.log('=== Basic Text Generation ===\n');

  // Create provider with TinyLlama model
  const llm = new WllamaProvider({
    modelUrl:
      'https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    modelConfig: {
      n_ctx: 2048,
      n_batch: 512,
      n_threads: 1,
    },
    progressCallback: ({ loaded, total }) => {
      const percent = ((loaded / total) * 100).toFixed(1);
      console.log(`Loading model: ${percent}%`);
    },
  });

  try {
    // Initialize and load model
    console.log('Initializing LLM...');
    await llm.initialize();
    console.log('Model loaded successfully!\n');

    // Generate text
    const prompt = 'What is the capital of France?';
    console.log(`Prompt: ${prompt}`);
    console.log('Generating response...\n');

    const response = await llm.generate(prompt, {
      maxTokens: 100,
      temperature: 0.7,
      topP: 0.9,
    });

    console.log(`Response: ${response}\n`);
  } catch (error) {
    console.error('Error:', error);
  } finally {
    // Cleanup
    await llm.dispose();
    console.log('Resources cleaned up');
  }
}

async function streamingGeneration() {
  console.log('=== Streaming Text Generation ===\n');

  const llm = new WllamaProvider({
    modelUrl:
      'https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    modelConfig: {
      n_ctx: 2048,
    },
  });

  try {
    await llm.initialize();

    const prompt = 'Tell me a short story about a robot learning to paint.';
    console.log(`Prompt: ${prompt}`);
    console.log('Streaming response:\n');

    // Stream tokens as they are generated
    for await (const chunk of llm.generateStream(prompt, {
      maxTokens: 200,
      temperature: 0.8,
    })) {
      process.stdout.write(chunk);
    }

    console.log('\n\nStreaming complete!');
  } catch (error) {
    console.error('Error:', error);
  } finally {
    await llm.dispose();
  }
}

async function ragExample() {
  console.log('=== RAG Pipeline Example ===\n');

  const llm = new WllamaProvider({
    modelUrl:
      'https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  });

  try {
    await llm.initialize();

    // Simulated retrieved context from vector database
    const context = `
Vector databases store high-dimensional vectors and enable similarity search.
They are commonly used for semantic search, recommendation systems, and RAG pipelines.
Popular vector databases include Pinecone, Weaviate, and Qdrant.
    `.trim();

    const question = 'What are vector databases used for?';

    // Build RAG prompt
    const prompt = `Context:\n${context}\n\nQuestion: ${question}\n\nAnswer based on the context:`;

    console.log('Question:', question);
    console.log('\nGenerating answer from context...\n');

    const answer = await llm.generate(prompt, {
      maxTokens: 150,
      temperature: 0.7,
    });

    console.log('Answer:', answer);
  } catch (error) {
    console.error('Error:', error);
  } finally {
    await llm.dispose();
  }
}

async function errorHandling() {
  console.log('=== Error Handling Example ===\n');

  // Example 1: Generate before initialization
  const llm1 = new WllamaProvider({
    modelUrl: 'https://example.com/model.gguf',
  });

  try {
    await llm1.generate('test');
  } catch (error) {
    console.log('Expected error (not initialized):', error.message);
  }

  // Example 2: Invalid model URL
  const llm2 = new WllamaProvider({
    modelUrl: 'https://invalid-url.com/nonexistent.gguf',
  });

  try {
    await llm2.initialize();
  } catch (error) {
    console.log('Expected error (invalid URL):', error.message);
  }

  // Example 3: Proper cleanup
  const llm3 = new WllamaProvider({
    modelUrl: 'https://example.com/model.gguf',
  });

  await llm3.dispose(); // Should not throw
  console.log('Cleanup successful even without initialization');
}

async function customConfiguration() {
  console.log('=== Custom Configuration Example ===\n');

  const llm = new WllamaProvider({
    modelUrl:
      'https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    modelConfig: {
      n_ctx: 4096, // Larger context window
      n_batch: 1024, // Larger batch size
      n_threads: 1,
      embeddings: false,
    },
    progressCallback: ({ loaded, total }) => {
      const percent = ((loaded / total) * 100).toFixed(1);
      const bar = '█'.repeat(Math.floor(Number(percent) / 2));
      console.log(`[${bar.padEnd(50)}] ${percent}%`);
    },
    wasmPaths: {
      // Custom WASM paths (optional)
      'single-thread/wllama.wasm': '/custom/path/wllama.wasm',
    },
  });

  console.log('Model info:', llm.getModelInfo());
  console.log('Initialized:', llm.isInitialized());

  // Note: Would need to actually initialize to use
  await llm.dispose();
}

// Run examples
async function main() {
  const examples = [
    { name: 'Basic Generation', fn: basicGeneration },
    { name: 'Streaming Generation', fn: streamingGeneration },
    { name: 'RAG Example', fn: ragExample },
    { name: 'Error Handling', fn: errorHandling },
    { name: 'Custom Configuration', fn: customConfiguration },
  ];

  console.log('WllamaProvider Examples\n');
  console.log('Note: These examples require actual model files to run.');
  console.log('Some examples are demonstrations of the API only.\n');

  for (const example of examples) {
    try {
      console.log(`\n${'='.repeat(60)}`);
      console.log(`Running: ${example.name}`);
      console.log('='.repeat(60));
      await example.fn();
    } catch (error) {
      console.error(`Error in ${example.name}:`, error);
    }
  }
}

// Uncomment to run
// main().catch(console.error);

export { basicGeneration, streamingGeneration, ragExample, errorHandling, customConfiguration };
