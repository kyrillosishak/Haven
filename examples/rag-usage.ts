/**
 * RAG Pipeline Usage Example
 * 
 * This example demonstrates how to use the RAG (Retrieval-Augmented Generation)
 * pipeline to build a question-answering system with local LLM inference.
 */

import { VectorDB, RAGPipelineManager, WllamaProvider } from '../src/index';

async function main() {
  console.log('=== RAG Pipeline Example ===\n');

  // Step 1: Initialize VectorDB
  console.log('1. Initializing VectorDB...');
  const vectorDB = new VectorDB({
    storage: { dbName: 'rag-example-db' },
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
  });

  await vectorDB.initialize();
  console.log('✓ VectorDB initialized\n');

  // Step 2: Insert sample documents
  console.log('2. Inserting sample documents...');
  const documents = [
    {
      text: 'Paris is the capital and most populous city of France. It is located in the north-central part of the country.',
      metadata: {
        title: 'Paris Overview',
        category: 'geography',
        source: 'encyclopedia',
      },
    },
    {
      text: 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. It was built in 1889 and is one of the most recognizable structures in the world.',
      metadata: {
        title: 'Eiffel Tower',
        category: 'landmarks',
        source: 'encyclopedia',
      },
    },
    {
      text: 'The Louvre Museum is the world\'s largest art museum and a historic monument in Paris. It is home to thousands of works of art, including the Mona Lisa.',
      metadata: {
        title: 'Louvre Museum',
        category: 'landmarks',
        source: 'encyclopedia',
      },
    },
    {
      text: 'French cuisine is renowned worldwide for its quality and diversity. Paris is home to numerous Michelin-starred restaurants.',
      metadata: {
        title: 'French Cuisine',
        category: 'culture',
        source: 'encyclopedia',
      },
    },
    {
      text: 'The Seine is a river that flows through Paris. It divides the city into the Left Bank and Right Bank.',
      metadata: {
        title: 'Seine River',
        category: 'geography',
        source: 'encyclopedia',
      },
    },
  ];

  const ids = await vectorDB.insertBatch(documents);
  console.log(`✓ Inserted ${ids.length} documents\n`);

  // Step 3: Initialize LLM Provider
  console.log('3. Initializing LLM Provider...');
  console.log('Note: This example uses WllamaProvider. You need to provide a valid model URL.');
  console.log('For testing, you can use a small model like TinyLlama or Phi-2.\n');

  // Example model URLs (you need to use actual URLs):
  // - TinyLlama: https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-q4_0.gguf
  // - Phi-2: https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf

  const llmProvider = new WllamaProvider({
    modelUrl: 'YOUR_MODEL_URL_HERE', // Replace with actual model URL
    modelConfig: {
      n_ctx: 2048,
      n_threads: 4,
    },
    progressCallback: ({ loaded, total }) => {
      const percent = ((loaded / total) * 100).toFixed(1);
      console.log(`  Loading model: ${percent}%`);
    },
  });

  // Uncomment to actually initialize (requires valid model URL)
  // await llmProvider.initialize();
  // console.log('✓ LLM Provider initialized\n');

  // For this example, we'll skip actual LLM initialization
  console.log('⚠ Skipping LLM initialization (requires valid model URL)\n');

  // Step 4: Create RAG Pipeline
  console.log('4. Creating RAG Pipeline...');
  
  // Access the internal embedding generator from VectorDB
  const embeddingGenerator = (vectorDB as any).embeddingGenerator;

  const ragPipeline = new RAGPipelineManager({
    vectorDB,
    llmProvider,
    embeddingGenerator,
    defaultContextTemplate: `Document {index}: {title}
{content}
(Category: {metadata.category}, Relevance: {score})`,
    defaultMaxContextTokens: 1500,
  });

  console.log('✓ RAG Pipeline created\n');

  // Step 5: Demonstrate RAG Query (without actual LLM)
  console.log('5. Demonstrating RAG Query Flow...\n');

  // We'll demonstrate the retrieval part
  const query = 'What are some famous landmarks in Paris?';
  console.log(`Query: "${query}"\n`);

  // Manually perform retrieval to show the process
  const queryVector = await embeddingGenerator.embed(query);
  const searchResults = await vectorDB.search({
    vector: queryVector,
    k: 3,
    filter: { field: 'category', operator: 'eq', value: 'landmarks' },
  });

  console.log('Retrieved Documents:');
  searchResults.forEach((result, index) => {
    console.log(`\n${index + 1}. ${result.metadata.title} (Score: ${result.score.toFixed(4)})`);
    console.log(`   ${result.metadata.content}`);
  });

  // Step 6: Demonstrate Streaming (conceptual)
  console.log('\n\n6. RAG Query with Streaming (conceptual)...\n');
  console.log('With a real LLM, you would use:');
  console.log(`
for await (const chunk of ragPipeline.queryStream('${query}')) {
  if (chunk.type === 'retrieval') {
    console.log('Sources retrieved:', chunk.sources.length);
  } else if (chunk.type === 'generation') {
    process.stdout.write(chunk.content);
  } else if (chunk.type === 'complete') {
    console.log('\\nDone!');
  }
}
  `);

  // Step 7: Demonstrate Custom Templates
  console.log('\n7. Custom Context Templates...\n');

  ragPipeline.setContextTemplate('[{index}] {title}: {content}');
  console.log('✓ Set custom template: "[{index}] {title}: {content}"');

  // Step 8: Configuration
  console.log('\n8. Pipeline Configuration...\n');
  const config = ragPipeline.getConfig();
  console.log('Current configuration:');
  console.log(`  - Context Template: ${config.defaultContextTemplate}`);
  console.log(`  - Max Context Tokens: ${config.defaultMaxContextTokens}`);

  // Step 9: Cleanup
  console.log('\n9. Cleanup...');
  await vectorDB.dispose();
  await llmProvider.dispose();
  console.log('✓ Resources cleaned up\n');

  console.log('=== Example Complete ===');
  console.log('\nTo use this example with a real LLM:');
  console.log('1. Replace YOUR_MODEL_URL_HERE with a valid GGUF model URL');
  console.log('2. Uncomment the llmProvider.initialize() line');
  console.log('3. Use ragPipeline.query() or ragPipeline.queryStream() for actual generation');
}

// Run the example
main().catch(console.error);
