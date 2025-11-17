/**
 * Standalone MCP Server Example
 * 
 * This example demonstrates how to set up a standalone MCP server
 * that can be used with AI assistants like Claude, ChatGPT, or other
 * MCP-compatible applications.
 */

import { VectorDB, MCPServer, RAGPipelineManager, WllamaProvider } from '../src/index';

/**
 * MCP Server Configuration
 */
interface MCPServerConfig {
  dbName: string;
  embeddingModel: string;
  llmModelUrl?: string;
  port?: number;
}

/**
 * Standalone MCP Server Class
 */
class StandaloneMCPServer {
  private vectorDB!: VectorDB;
  private mcpServer!: MCPServer;
  private ragPipeline?: RAGPipelineManager;
  private config: MCPServerConfig;

  constructor(config: MCPServerConfig) {
    this.config = config;
  }

  /**
   * Initialize the server
   */
  async initialize(): Promise<void> {
    console.log('🚀 Initializing MCP Server...\n');

    // Initialize VectorDB
    console.log('1. Setting up VectorDB...');
    this.vectorDB = new VectorDB({
      storage: {
        dbName: this.config.dbName,
        version: 1,
      },
      index: {
        dimensions: 384,
        metric: 'cosine',
        indexType: 'kdtree',
      },
      embedding: {
        model: this.config.embeddingModel,
        device: 'wasm',
        cache: true,
      },
    });

    await this.vectorDB.initialize();
    console.log('✓ VectorDB initialized\n');

    // Initialize LLM (optional)
    if (this.config.llmModelUrl) {
      console.log('2. Setting up LLM Provider...');
      const llmProvider = new WllamaProvider({
        modelUrl: this.config.llmModelUrl,
        modelConfig: {
          n_ctx: 2048,
          n_threads: 4,
        },
        progressCallback: ({ loaded, total }) => {
          const percent = ((loaded / total) * 100).toFixed(1);
          process.stdout.write(`\r  Loading model: ${percent}%`);
        },
      });

      await llmProvider.initialize();
      console.log('\n✓ LLM Provider initialized\n');

      // Initialize RAG Pipeline
      this.ragPipeline = new RAGPipelineManager({
        vectorDB: this.vectorDB,
        llmProvider,
        embeddingGenerator: (this.vectorDB as any).embeddingGenerator,
      });
    }

    // Initialize MCP Server
    console.log('3. Setting up MCP Server...');
    this.mcpServer = new MCPServer({
      vectorDB: this.vectorDB,
      ragPipeline: this.ragPipeline,
    });
    console.log('✓ MCP Server initialized\n');

    this.printServerInfo();
  }

  /**
   * Print server information
   */
  private printServerInfo(): void {
    console.log('═'.repeat(60));
    console.log('MCP Server Ready');
    console.log('═'.repeat(60));
    console.log(`Database: ${this.config.dbName}`);
    console.log(`Embedding Model: ${this.config.embeddingModel}`);
    console.log(`LLM: ${this.config.llmModelUrl ? 'Enabled' : 'Disabled'}`);
    console.log(`\nAvailable Tools: ${this.mcpServer.getToolNames().join(', ')}`);
    console.log('═'.repeat(60));
    console.log();
  }

  /**
   * Handle tool execution
   */
  async executeTool(toolName: string, params: any): Promise<any> {
    console.log(`\n📞 Tool Call: ${toolName}`);
    console.log(`Parameters:`, JSON.stringify(params, null, 2));

    try {
      const result = await this.mcpServer.executeTool(toolName, params);
      console.log(`✓ Success`);
      return result;
    } catch (error: any) {
      console.error(`✗ Error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get available tools
   */
  getTools() {
    return this.mcpServer.getTools();
  }

  /**
   * Shutdown the server
   */
  async shutdown(): Promise<void> {
    console.log('\n🛑 Shutting down MCP Server...');
    await this.vectorDB.dispose();
    console.log('✓ Server shutdown complete');
  }
}

/**
 * Example usage and testing
 */
async function main() {
  console.log('=== Standalone MCP Server Example ===\n');

  // Create and initialize server
  const server = new StandaloneMCPServer({
    dbName: 'mcp-server-db',
    embeddingModel: 'Xenova/all-MiniLM-L6-v2',
    // llmModelUrl: 'https://huggingface.co/...', // Optional
  });

  await server.initialize();

  // Example 1: Insert documents
  console.log('\n📝 Example 1: Inserting Documents\n');

  const documents = [
    {
      content: 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel.',
      metadata: {
        title: 'Eiffel Tower',
        category: 'landmarks',
        location: 'Paris, France',
      },
    },
    {
      content: 'The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, and other materials. It was built to protect Chinese states from invasions.',
      metadata: {
        title: 'Great Wall of China',
        category: 'landmarks',
        location: 'China',
      },
    },
    {
      content: 'The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor. It was a gift from France to the United States.',
      metadata: {
        title: 'Statue of Liberty',
        category: 'landmarks',
        location: 'New York, USA',
      },
    },
  ];

  for (const doc of documents) {
    const result = await server.executeTool('insert_document', doc);
    console.log(`  Inserted: ${doc.metadata.title} (ID: ${result.id})`);
  }

  // Example 2: Search vectors
  console.log('\n\n🔍 Example 2: Searching Documents\n');

  const searchResult = await server.executeTool('search_vectors', {
    query: 'famous towers and monuments',
    k: 3,
  });

  console.log(`\nFound ${searchResult.count} results:`);
  searchResult.results.forEach((result: any, index: number) => {
    console.log(`\n${index + 1}. ${result.metadata.title}`);
    console.log(`   Location: ${result.metadata.location}`);
    console.log(`   Score: ${(result.score * 100).toFixed(1)}%`);
    console.log(`   Content: ${result.metadata.content.substring(0, 80)}...`);
  });

  // Example 3: Filtered search
  console.log('\n\n🎯 Example 3: Filtered Search\n');

  const filteredResult = await server.executeTool('search_vectors', {
    query: 'historical structures',
    k: 5,
    filter: {
      field: 'category',
      operator: 'eq',
      value: 'landmarks',
    },
  });

  console.log(`\nFound ${filteredResult.count} landmarks:`);
  filteredResult.results.forEach((result: any, index: number) => {
    console.log(`  ${index + 1}. ${result.metadata.title} - ${result.metadata.location}`);
  });

  // Example 4: List all tools
  console.log('\n\n🛠️  Example 4: Available Tools\n');

  const tools = server.getTools();
  tools.forEach((tool) => {
    console.log(`\n${tool.name}`);
    console.log(`  Description: ${tool.description}`);
    console.log(`  Input Schema:`, JSON.stringify(tool.inputSchema, null, 2));
  });

  // Example 5: Delete document
  console.log('\n\n🗑️  Example 5: Deleting Document\n');

  const deleteResult = await server.executeTool('delete_document', {
    id: searchResult.results[0].id,
  });

  if (deleteResult.success) {
    console.log(`  Deleted: ${deleteResult.id}`);
  }

  // Example 6: RAG Query (if LLM is available)
  console.log('\n\n💬 Example 6: RAG Query\n');

  try {
    const ragResult = await server.executeTool('rag_query', {
      query: 'Tell me about famous landmarks in France',
      topK: 2,
      maxTokens: 150,
      temperature: 0.7,
    });

    console.log('\nAnswer:', ragResult.answer);
    console.log(`\nSources: ${ragResult.sources.length} documents`);
    console.log(`Retrieval time: ${ragResult.metadata.retrievalTime}ms`);
    console.log(`Generation time: ${ragResult.metadata.generationTime}ms`);
  } catch (error: any) {
    console.log('  Note: RAG query requires LLM to be configured');
    console.log(`  Error: ${error.message}`);
  }

  // Example 7: Integration with MCP clients
  console.log('\n\n🔌 Example 7: MCP Client Integration\n');

  console.log('To use this server with MCP-compatible clients:\n');
  console.log('1. Claude Desktop:');
  console.log('   Add to claude_desktop_config.json:');
  console.log(`   {
     "mcpServers": {
       "vectordb": {
         "command": "node",
         "args": ["path/to/mcp-server-standalone.js"]
       }
     }
   }`);

  console.log('\n2. Custom Integration:');
  console.log(`   const server = new StandaloneMCPServer({ ... });
   await server.initialize();
   
   // Execute tools
   const result = await server.executeTool('search_vectors', {
     query: 'your query',
     k: 5
   });`);

  console.log('\n3. HTTP API Wrapper:');
  console.log('   Wrap the server in an HTTP API for remote access');

  // Example 8: Error handling
  console.log('\n\n⚠️  Example 8: Error Handling\n');

  try {
    await server.executeTool('search_vectors', {
      // Missing required 'query' parameter
      k: 5,
    });
  } catch (error: any) {
    console.log(`  ✓ Caught validation error: ${error.message}`);
  }

  try {
    await server.executeTool('unknown_tool', {});
  } catch (error: any) {
    console.log(`  ✓ Caught unknown tool error: ${error.message}`);
  }

  // Example 9: Performance monitoring
  console.log('\n\n📊 Example 9: Performance Monitoring\n');

  const iterations = 10;
  const times: number[] = [];

  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await server.executeTool('search_vectors', {
      query: 'test query',
      k: 3,
    });
    times.push(performance.now() - start);
  }

  const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);

  console.log(`  Average search time: ${avgTime.toFixed(2)}ms`);
  console.log(`  Min: ${minTime.toFixed(2)}ms, Max: ${maxTime.toFixed(2)}ms`);

  // Example 10: Best practices
  console.log('\n\n✨ Example 10: Best Practices\n');

  console.log('  ✓ Initialize server once and reuse');
  console.log('  ✓ Handle errors gracefully');
  console.log('  ✓ Validate input parameters');
  console.log('  ✓ Monitor performance metrics');
  console.log('  ✓ Implement proper shutdown procedures');
  console.log('  ✓ Use metadata filtering for better results');
  console.log('  ✓ Cache embeddings when possible');
  console.log('  ✓ Set appropriate context limits for RAG');

  // Shutdown
  await server.shutdown();

  console.log('\n=== MCP Server Example Complete ===');
}

// Run the example
if (require.main === module) {
  main().catch(console.error);
}

export { StandaloneMCPServer, MCPServerConfig };
