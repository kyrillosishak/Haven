/**
 * Example: Using WebLLMProvider for WebGPU-accelerated LLM inference
 */

import { WebLLMProvider } from '../src/llm/WebLLMProvider';

async function checkWebGPUAvailability() {
  console.log('=== Checking WebGPU Availability ===\n');

  const available = await WebLLMProvider.isWebGPUAvailable();
  console.log(`WebGPU Available: ${available}`);

  if (!available) {
    console.log('\nWebGPU is not available in this environment.');
    console.log('WebLLM requires WebGPU support (Chrome 113+, Edge 113+).');
    console.log('Consider using WllamaProvider as a fallback.\n');
  } else {
    console.log('\nWebGPU is available! You can use WebLLMProvider.\n');
  }

  return available;
}

async function basicGeneration() {
  console.log('=== Basic Text Generation ===\n');

  // Create provider with Llama 3.2 1B model
  const llm = new WebLLMProvider({
    model: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
    engineConfig: {
      initProgressCallback: ({ progress, text }) => {
        console.log(`${text}: ${(progress * 100).toFixed(1)}%`);
      },
      logLevel: 'ERROR',
    },
    chatConfig: {
      temperature: 0.7,
      top_p: 0.9,
      max_tokens: 512,
    },
  });

  try {
    // Initialize and load model
    console.log('Initializing WebLLM...');
    await llm.initialize();
    console.log('Model loaded successfully!\n');

    // Get model info
    const info = llm.getModelInfo();
    console.log('Model Info:', info);

    // Generate text
    const prompt = 'What is the capital of France?';
    console.log(`\nPrompt: ${prompt}`);
    console.log('Generating response...\n');

    const response = await llm.generate(prompt, {
      maxTokens: 100,
      temperature: 0.7,
    });

    console.log(`Response: ${response}\n`);

    // Get runtime statistics
    const stats = await llm.getRuntimeStats();
    if (stats) {
      console.log('Runtime Stats:', stats);
    }
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

  const llm = new WebLLMProvider({
    model: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
    engineConfig: {
      logLevel: 'ERROR',
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

async function multiTurnConversation() {
  console.log('=== Multi-Turn Conversation ===\n');

  const llm = new WebLLMProvider({
    model: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
  });

  try {
    await llm.initialize();

    // First turn
    console.log('User: What is the capital of France?');
    const response1 = await llm.generate('What is the capital of France?', {
      maxTokens: 50,
    });
    console.log(`Assistant: ${response1}\n`);

    // Second turn (context is maintained)
    console.log('User: What is its population?');
    const response2 = await llm.generate('What is its population?', {
      maxTokens: 50,
    });
    console.log(`Assistant: ${response2}\n`);

    // Third turn
    console.log('User: What are some famous landmarks there?');
    const response3 = await llm.generate('What are some famous landmarks there?', {
      maxTokens: 100,
    });
    console.log(`Assistant: ${response3}\n`);

    // Reset conversation
    console.log('Resetting conversation...');
    await llm.resetChat();
    console.log('Conversation reset!\n');

    // New conversation
    console.log('User: Hello!');
    const response4 = await llm.generate('Hello!', {
      maxTokens: 50,
    });
    console.log(`Assistant: ${response4}\n`);
  } catch (error) {
    console.error('Error:', error);
  } finally {
    await llm.dispose();
  }
}

async function ragExample() {
  console.log('=== RAG Pipeline Example ===\n');

  const llm = new WebLLMProvider({
    model: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
  });

  try {
    await llm.initialize();

    // Simulated retrieved context from vector database
    const context = `
Vector databases store high-dimensional vectors and enable similarity search.
They are commonly used for semantic search, recommendation systems, and RAG pipelines.
Popular vector databases include Pinecone, Weaviate, and Qdrant.
Browser-based vector databases like this one enable offline-first AI applications.
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

async function gracefulDegradation() {
  console.log('=== Graceful Degradation Example ===\n');

  try {
    // Check WebGPU availability
    const webGPUAvailable = await WebLLMProvider.isWebGPUAvailable();

    if (webGPUAvailable) {
      console.log('WebGPU is available, using WebLLMProvider');
      const llm = new WebLLMProvider({
        model: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
      });
      await llm.initialize();
      console.log('WebLLMProvider initialized successfully');
      await llm.dispose();
    } else {
      console.log('WebGPU not available, would fallback to WllamaProvider');
      console.log('(WllamaProvider example not shown here)');
    }
  } catch (error) {
    if (error instanceof Error && error.message.includes('WebGPU')) {
      console.log('WebGPU initialization failed, fallback recommended');
      console.log('Error:', error.message);
    } else {
      console.error('Unexpected error:', error);
    }
  }
}

async function errorHandling() {
  console.log('=== Error Handling Example ===\n');

  // Example 1: Generate before initialization
  const llm1 = new WebLLMProvider({
    model: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
  });

  try {
    await llm1.generate('test');
  } catch (error) {
    console.log('Expected error (not initialized):', error.message);
  }

  // Example 2: Stream before initialization
  try {
    const generator = llm1.generateStream('test');
    await generator.next();
  } catch (error) {
    console.log('Expected error (streaming not initialized):', error.message);
  }

  // Example 3: Reset chat before initialization
  try {
    await llm1.resetChat();
  } catch (error) {
    console.log('Expected error (reset before init):', error.message);
  }

  // Example 4: Proper cleanup
  await llm1.dispose(); // Should not throw
  console.log('Cleanup successful even without initialization');
}

async function customConfiguration() {
  console.log('=== Custom Configuration Example ===\n');

  const llm = new WebLLMProvider({
    model: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC', // Smaller model
    engineConfig: {
      initProgressCallback: ({ progress, text }) => {
        const percent = (progress * 100).toFixed(1);
        const bar = '█'.repeat(Math.floor(progress * 50));
        console.log(`[${bar.padEnd(50)}] ${percent}% - ${text}`);
      },
      logLevel: 'INFO',
    },
    chatConfig: {
      temperature: 0.5,
      top_p: 0.8,
      max_tokens: 256,
      frequency_penalty: 0.5,
      presence_penalty: 0.5,
    },
  });

  console.log('Model info:', llm.getModelInfo());
  console.log('Initialized:', llm.isInitialized());

  // Note: Would need to actually initialize to use
  await llm.dispose();
}

async function performanceComparison() {
  console.log('=== Performance Comparison ===\n');

  const llm = new WebLLMProvider({
    model: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
  });

  try {
    await llm.initialize();

    const prompt = 'Count from 1 to 10.';
    console.log(`Prompt: ${prompt}\n`);

    // Measure generation time
    console.log('Generating with timing...');
    const startTime = performance.now();

    const response = await llm.generate(prompt, {
      maxTokens: 50,
    });

    const endTime = performance.now();
    const duration = ((endTime - startTime) / 1000).toFixed(2);

    console.log(`Response: ${response}`);
    console.log(`\nGeneration time: ${duration}s`);

    // Get runtime stats
    const stats = await llm.getRuntimeStats();
    if (stats) {
      console.log(`Runtime stats: ${stats}`);
    }
  } catch (error) {
    console.error('Error:', error);
  } finally {
    await llm.dispose();
  }
}

// Run examples
async function main() {
  const examples = [
    { name: 'Check WebGPU Availability', fn: checkWebGPUAvailability },
    { name: 'Basic Generation', fn: basicGeneration },
    { name: 'Streaming Generation', fn: streamingGeneration },
    { name: 'Multi-Turn Conversation', fn: multiTurnConversation },
    { name: 'RAG Example', fn: ragExample },
    { name: 'Graceful Degradation', fn: gracefulDegradation },
    { name: 'Error Handling', fn: errorHandling },
    { name: 'Custom Configuration', fn: customConfiguration },
    { name: 'Performance Comparison', fn: performanceComparison },
  ];

  console.log('WebLLMProvider Examples\n');
  console.log('Note: These examples require WebGPU support (Chrome 113+, Edge 113+).');
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

export {
  checkWebGPUAvailability,
  basicGeneration,
  streamingGeneration,
  multiTurnConversation,
  ragExample,
  gracefulDegradation,
  errorHandling,
  customConfiguration,
  performanceComparison,
};
