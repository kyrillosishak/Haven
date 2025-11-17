# Browser VectorDB Examples

This directory contains comprehensive examples demonstrating various features and use cases of the Browser VectorDB library.

## 📚 Available Examples

### 1. Embedding Generation (`embedding-usage.ts`)
**Difficulty:** Beginner  
**Topics:** Embeddings, Transformers.js

Explore text embedding generation:
- Loading embedding models
- Generating embeddings for text
- Batch embedding operations
- Model caching strategies

```bash
node dist/examples/embedding-usage.js
```

### 2. RAG Pipeline (`rag-usage.ts`)
**Difficulty:** Intermediate  
**Topics:** RAG, LLM Integration, Context Management

Build retrieval-augmented generation workflows:
- Setting up RAG pipelines
- Document insertion and retrieval
- Context formatting
- LLM integration (wllama/WebLLM)
- Streaming responses

```bash
node dist/examples/rag-usage.js
```

### 3. LLM Integration (`llm-usage.ts`)
**Difficulty:** Intermediate  
**Topics:** Local LLMs, Text Generation

Work with local language models:
- wllama (WASM-based inference)
- WebLLM (WebGPU-accelerated)
- Text generation
- Streaming completions

```bash
node dist/examples/llm-usage.js
```

### 4. WebLLM Usage (`webllm-usage.ts`)
**Difficulty:** Intermediate  
**Topics:** WebGPU, Accelerated Inference

Leverage WebGPU for fast inference:
- WebGPU setup and detection
- Model loading and initialization
- Chat completions
- Performance optimization

```bash
node dist/examples/webllm-usage.js
```

### 5. MCP Server (`mcp-usage.ts`)
**Difficulty:** Intermediate  
**Topics:** MCP Protocol, Tool Integration

Integrate with AI assistants using MCP:
- MCP tool definitions
- Tool execution
- Parameter validation
- Error handling

```bash
node dist/examples/mcp-usage.js
```

### 6. Standalone MCP Server (`mcp-server-standalone.ts`)
**Difficulty:** Advanced  
**Topics:** Server Setup, Production Deployment

Build a production-ready MCP server:
- Server initialization and configuration
- Tool management
- Performance monitoring
- Integration patterns
- Best practices

```bash
node dist/examples/mcp-server-standalone.js
```

### 7. Performance Optimization (`performance-usage.ts`)
**Difficulty:** Advanced  
**Topics:** Caching, Memory Management, Optimization

Optimize performance for production:
- LRU caching strategies
- Memory management
- Batch operations
- Progressive loading
- Performance metrics

```bash
node dist/examples/performance-usage.js
```

### 8. Multimodal Search (`multimodal-search.ts`)
**Difficulty:** Advanced  
**Topics:** CLIP, Image Embeddings, Cross-Modal Search

Implement text and image search:
- CLIP model integration
- Text-to-image search
- Image-to-text search
- Cross-modal retrieval
- Multimodal filtering

```bash
node dist/examples/multimodal-search.js
```

### 9. Document Q&A (`document-qa.ts`)
**Difficulty:** Advanced  
**Topics:** Document Processing, Q&A Systems, RAG

Build intelligent document Q&A systems:
- Document chunking strategies
- Metadata management
- Citation tracking
- Confidence scoring
- Multi-document search

```bash
node dist/examples/document-qa.js
```

### 10. Performance Benchmarking (`benchmark-usage.ts`)
**Difficulty:** Advanced  
**Topics:** Benchmarking, Performance Testing, Metrics

Run comprehensive performance benchmarks:
- Search latency across dataset sizes
- Insertion throughput measurement
- Memory usage profiling
- Cache performance analysis
- Model load time testing
- Cross-browser comparison

```bash
node dist/examples/benchmark-usage.js
```

## 🎨 Interactive Demos

### Semantic Search Demo (`semantic-search-demo.html`)
**Type:** Web Application  
**Features:** Beautiful UI, Real-time Search, Filtering

A complete semantic search application with:
- Modern, responsive UI
- Real-time search with filters
- Document management
- Statistics dashboard
- Export/import functionality

**To run:**
```bash
# Serve the HTML file with a local server
npx serve examples
# Open http://localhost:3000/semantic-search-demo.html
```

### RAG Chatbot Demo (`rag-chatbot-demo.html`)
**Type:** Web Application  
**Features:** Chat Interface, Streaming, Source Citations

An interactive chatbot powered by RAG:
- Chat-style interface
- Streaming responses
- Source attribution
- Conversation history
- Customizable settings

**To run:**
```bash
npx serve examples
# Open http://localhost:3000/rag-chatbot-demo.html
```

### Benchmark Demo - Simple (`benchmark-demo-simple.html`)
**Type:** Web Application  
**Features:** Standalone Performance Testing

Lightweight performance benchmarking without dependencies:
- Tests basic JavaScript operations
- Array and object performance
- IndexedDB operations
- Environment detection
- No build required

**To run:**
```bash
npx serve examples
# Open http://localhost:3000/benchmark-demo-simple.html
```

### Export/Import Demo (`export-import-usage.ts`)
**Type:** Node.js Script  
**Features:** Data Portability, Backup/Restore

Export and import database data:
- Export to JSON
- Import from JSON
- Backup and restore workflows

**To run:**
```bash
node dist/examples/export-import-usage.js
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
npm install

# Build the library
npm run build
```

### Running Examples

All TypeScript examples need to be compiled first:

```bash
# Build the library
npm run build

# Run any example
node dist/examples/<example-name>.js
```

### Using in Browser

For HTML demos, you can use any static file server:

```bash
# Option 1: Using npx serve
npx serve examples

# Option 2: Using Python
python -m http.server 8000

# Option 3: Using Node.js http-server
npx http-server examples
```

## 📖 Learning Path

### Beginner Path
1. Read `docs/QUICKSTART.md` to understand initialization
2. Explore `embedding-usage.ts` to learn about embeddings
3. Try `semantic-search-demo.html` for a visual understanding

### Intermediate Path
1. Learn RAG with `rag-usage.ts`
2. Explore LLM integration with `llm-usage.ts`
3. Try `rag-chatbot-demo.html` for interactive RAG
4. Study MCP integration with `mcp-usage.ts`

### Advanced Path
1. Master performance with `performance-usage.ts`
2. Build multimodal apps with `multimodal-search.ts`
3. Create Q&A systems with `document-qa.ts`
4. Deploy with `mcp-server-standalone.ts`

## 🎯 Use Case Examples

### Semantic Search Application
```typescript
// See: semantic-search-demo.html
// Features: Search, filtering, metadata, persistence
```

### Chatbot with RAG
```typescript
// See: rag-chatbot-demo.html, rag-usage.ts
// Features: Context retrieval, LLM generation, citations
```

### Document Q&A System
```typescript
// See: document-qa.ts
// Features: Document chunking, Q&A, citations, confidence
```

### Image Search
```typescript
// See: multimodal-search.ts
// Features: Text-to-image, image-to-image, CLIP embeddings
```

### AI Agent Integration
```typescript
// See: mcp-server-standalone.ts, mcp-usage.ts
// Features: MCP protocol, tool execution, AI assistant integration
```

## 🔧 Configuration Examples

### Basic Configuration
```typescript
const db = new VectorDB({
  storage: { dbName: 'my-app' },
  index: { dimensions: 384, metric: 'cosine' },
  embedding: { model: 'Xenova/all-MiniLM-L6-v2', device: 'wasm' },
});
```

### With LLM
```typescript
const db = new VectorDB({
  storage: { dbName: 'my-app' },
  index: { dimensions: 384, metric: 'cosine' },
  embedding: { model: 'Xenova/all-MiniLM-L6-v2', device: 'wasm' },
  llm: {
    provider: 'wllama',
    model: 'https://huggingface.co/.../model.gguf',
  },
});
```

### Performance Optimized
```typescript
const db = new VectorDB({
  storage: { dbName: 'my-app', maxVectors: 100000 },
  index: { 
    dimensions: 384, 
    metric: 'cosine',
    indexType: 'hnsw', // Faster for large datasets
  },
  embedding: { 
    model: 'Xenova/all-MiniLM-L6-v2', 
    device: 'webgpu', // Use GPU if available
    cache: true,
  },
});
```

## 📊 Performance Tips

1. **Use Batch Operations**: Insert multiple documents at once
   ```typescript
   await db.insertBatch(documents); // Faster than individual inserts
   ```

2. **Enable Caching**: Cache embeddings and models
   ```typescript
   embedding: { cache: true }
   ```

3. **Optimize Index**: Choose the right index type
   ```typescript
   index: { indexType: 'hnsw' } // Better for large datasets
   ```

4. **Use WebGPU**: Enable GPU acceleration when available
   ```typescript
   embedding: { device: 'webgpu' }
   ```

5. **Filter Early**: Use metadata filters to reduce search space
   ```typescript
   await db.search({ 
     text: query, 
     k: 10,
     filter: { field: 'category', operator: 'eq', value: 'tech' }
   });
   ```

## 🐛 Troubleshooting

### Model Loading Issues
```typescript
// Check if model is cached
embedding: { cache: true }

// Use smaller models for testing
embedding: { model: 'Xenova/all-MiniLM-L6-v2' } // 23MB
```

### Memory Issues
```typescript
// Limit vector count
storage: { maxVectors: 10000 }

// Use progressive loading
// See: performance-usage.ts
```

### WebGPU Not Available
```typescript
// Fallback to WASM
embedding: { device: 'wasm' }
```

### IndexedDB Quota Exceeded
```typescript
// Export and clear old data
const data = await db.export();
await db.clear();
```

## 🤝 Contributing Examples

Want to add an example? Follow these guidelines:

1. **Clear Purpose**: Each example should demonstrate specific features
2. **Well Commented**: Explain what each section does
3. **Error Handling**: Show proper error handling patterns
4. **Best Practices**: Demonstrate recommended approaches
5. **Self-Contained**: Examples should run independently

## 📚 Additional Resources

- [Main Documentation](../README.md)
- [API Reference](../docs/API.md)
- [Performance Guide](../docs/PERFORMANCE.md)
- [Testing Guide](../docs/TESTING.md)

## 💡 Tips for Learning

1. **Start Simple**: Begin with basic examples and gradually move to advanced ones
2. **Experiment**: Modify examples to understand how things work
3. **Read Comments**: Examples are heavily commented for learning
4. **Check Console**: Many examples log detailed information
5. **Use Demos**: Interactive HTML demos help visualize concepts

## 🎓 Example Projects

Looking for complete project examples? Check out:

- **Semantic Search App**: `semantic-search-demo.html`
- **RAG Chatbot**: `rag-chatbot-demo.html`
- **MCP Server**: `mcp-server-standalone.ts`

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Full Docs](../README.md)

---

Happy coding! 🚀
