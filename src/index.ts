/**
 * Browser-based Vector Database
 * 
 * A production-quality vector database that runs entirely in the browser
 * with support for semantic search, RAG pipelines, and local LLM integration.
 */

// Core API
export { VectorDB } from './core/VectorDB';
export type { 
  VectorDBConfig, 
  StorageConfig, 
  IndexConfig, 
  EmbeddingConfig, 
  LLMConfig,
  InsertData,
  ExportData,
  ExportOptions,
  ImportOptions
} from './core/types';

// Storage
export type { VectorRecord, MetadataFilter, CompoundFilter, Filter, StorageManager } from './storage/types';
export { IndexedDBStorage } from './storage/IndexedDBStorage';

// Index
export type { SearchQuery, SearchResult, IndexStats } from './index/types';
export { IndexManager } from './index/IndexManager';
export type { IndexManagerConfig } from './index/IndexManager';

// Embedding
export type { EmbeddingGenerator } from './embedding/types';
export { TransformersEmbedding, type TransformersEmbeddingConfig } from './embedding/TransformersEmbedding';

// LLM
export type { LLMProvider, GenerateOptions } from './llm/types';
export { WllamaProvider, type WllamaProviderConfig } from './llm/WllamaProvider';

// RAG
export { RAGPipelineManager, type RAGPipelineConfig } from './rag/RAGPipelineManager';
export type { RAGPipeline, RAGOptions, RAGResult, RAGStreamChunk } from './rag/types';

// MCP
export { MCPServer, type MCPServerConfig } from './mcp/MCPServer';
export type { MCPTool, JSONSchema } from './mcp/types';

// Errors
export { 
  VectorDBError, 
  StorageQuotaError, 
  DimensionMismatchError, 
  ModelLoadError, 
  IndexCorruptedError,
  InputValidator,
  ErrorHandler,
  DEFAULT_RETRY_CONFIG,
  type RetryConfig
} from './errors';

// Performance
export { 
  LRUCache,
  MemoryManager,
  WorkerPool,
  ProgressiveLoader,
  BatchOptimizer,
  PerformanceOptimizer,
  Benchmark,
  BenchmarkRunner
} from './performance';
export type {
  CacheEntry,
  LRUCacheConfig,
  MemoryManagerConfig,
  MemoryStats,
  WorkerTask,
  WorkerResponse,
  WorkerPoolConfig,
  ProgressiveLoaderConfig,
  LoadProgress,
  BatchOptimizerConfig,
  PendingOperation,
  PerformanceConfig,
  BenchmarkResult,
  BenchmarkEnvironment,
  BenchmarkSuite
} from './performance';

// Version
export const VERSION = '0.1.0';
