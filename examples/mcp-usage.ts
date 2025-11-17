/**
 * Example: Using the MCP Server for AI agent integration
 * 
 * This example demonstrates how to set up and use the MCP-compatible
 * interface for vector database operations.
 */

import { VectorDB, MCPServer, RAGPipelineManager, WllamaProvider } from '../src/index';

async function main() {
  console.log('=== MCP Server Example ===\n');

  // 1. Initialize VectorDB
  console.log('1. Initializing VectorDB...');
  const vectorDB = new VectorDB({
    storage: {
      dbName: 'mcp-example-db',
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
  });

  await vectorDB.initialize();
  console.log('✓ VectorDB initialized\n');

  // 2. Initialize RAG Pipeline (optional, for rag_query tool)
  console.log('2. Initializing RAG Pipeline...');
  const llmProvider = new WllamaProvider({
    modelUrl: 'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf',
    modelConfig: {
      n_ctx: 2048,
    },
  });

  await llmProvider.initialize();

  const ragPipeline = new RAGPipelineManager({
    vectorDB,
    llmProvider,
    embeddingGenerator: (vectorDB as any).embeddingGenerator,
  });
  console.log('✓ RAG Pipeline initialized\n');

  // 3. Create MCP Server
  console.log('3. Creating MCP Server...');
  const mcpServer = new MCPServer({
    vectorDB,
    ragPipeline,
  });
  console.log('✓ MCP Server created\n');

  // 4. List available tools
  console.log('4. Available MCP Tools:');
  const tools = mcpServer.getTools();
  for (const tool of tools) {
    console.log(`   - ${tool.name}: ${tool.description}`);
  }
  console.log();

  // 5. Use insert_document tool
  console.log('5. Inserting documents using MCP tool...');
  const doc1Result = await mcpServer.executeTool('insert_document', {
    content: 'The quick brown fox jumps over the lazy dog.',
    metadata: {
      title: 'Fox Story',
      category: 'animals',
      tags: ['fox', 'dog'],
    },
  });
  console.log(`   ✓ Inserted document: ${doc1Result.id}`);

  const doc2Result = await mcpServer.executeTool('insert_document', {
    content: 'Machine learning is a subset of artificial intelligence.',
    metadata: {
      title: 'ML Introduction',
      category: 'technology',
      tags: ['ml', 'ai'],
    },
  });
  console.log(`   ✓ Inserted document: ${doc2Result.id}`);

  const doc3Result = await mcpServer.executeTool('insert_document', {
    content: 'Vector databases enable semantic search capabilities.',
    metadata: {
      title: 'Vector DB Overview',
      category: 'technology',
      tags: ['database', 'search'],
    },
  });
  console.log(`   ✓ Inserted document: ${doc3Result.id}\n`);

  // 6. Use search_vectors tool
  console.log('6. Searching using MCP tool...');
  const searchResult = await mcpServer.executeTool('search_vectors', {
    query: 'artificial intelligence and machine learning',
    k: 2,
  });

  console.log(`   Found ${searchResult.count} results:`);
  for (const result of searchResult.results) {
    console.log(`   - [Score: ${result.score.toFixed(4)}] ${result.metadata.title}`);
    console.log(`     ${result.metadata.content}`);
  }
  console.log();

  // 7. Use search with filters
  console.log('7. Searching with metadata filter...');
  const filteredSearchResult = await mcpServer.executeTool('search_vectors', {
    query: 'information',
    k: 5,
    filter: {
      field: 'category',
      operator: 'eq',
      value: 'technology',
    },
  });

  console.log(`   Found ${filteredSearchResult.count} results in 'technology' category:`);
  for (const result of filteredSearchResult.results) {
    console.log(`   - ${result.metadata.title}`);
  }
  console.log();

  // 8. Use rag_query tool
  console.log('8. Executing RAG query using MCP tool...');
  try {
    const ragResult = await mcpServer.executeTool('rag_query', {
      query: 'What is machine learning?',
      topK: 2,
      maxTokens: 100,
      temperature: 0.7,
    });

    console.log('   Answer:', ragResult.answer);
    console.log(`   Sources: ${ragResult.sources.length} documents`);
    console.log(`   Retrieval time: ${ragResult.metadata.retrievalTime}ms`);
    console.log(`   Generation time: ${ragResult.metadata.generationTime}ms`);
  } catch (error) {
    console.log('   Note: RAG query requires LLM model to be loaded');
    console.log(`   Error: ${error}`);
  }
  console.log();

  // 9. Use delete_document tool
  console.log('9. Deleting document using MCP tool...');
  const deleteResult = await mcpServer.executeTool('delete_document', {
    id: doc1Result.id,
  });

  if (deleteResult.success) {
    console.log(`   ✓ Deleted document: ${deleteResult.id}`);
  }
  console.log();

  // 10. Tool introspection
  console.log('10. Tool Introspection:');
  console.log(`   Available tools: ${mcpServer.getToolNames().join(', ')}`);
  console.log(`   Has 'search_vectors': ${mcpServer.hasTool('search_vectors')}`);
  console.log(`   Has 'unknown_tool': ${mcpServer.hasTool('unknown_tool')}`);
  
  const searchTool = mcpServer.getTool('search_vectors');
  if (searchTool) {
    console.log(`   'search_vectors' schema:`, JSON.stringify(searchTool.inputSchema, null, 2));
  }
  console.log();

  // 11. Error handling
  console.log('11. Error Handling:');
  try {
    await mcpServer.executeTool('search_vectors', {
      // Missing required 'query' parameter
      k: 5,
    });
  } catch (error: any) {
    console.log(`   ✓ Caught validation error: ${error.message}`);
  }

  try {
    await mcpServer.executeTool('unknown_tool', {});
  } catch (error: any) {
    console.log(`   ✓ Caught unknown tool error: ${error.message}`);
  }
  console.log();

  // 12. Cleanup
  console.log('12. Cleaning up...');
  await llmProvider.dispose();
  await vectorDB.dispose();
  console.log('✓ Cleanup complete\n');

  console.log('=== MCP Server Example Complete ===');
}

// Run the example
main().catch(console.error);
