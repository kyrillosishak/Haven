/**
 * RAG (Retrieval-Augmented Generation) layer types
 */

import type { SearchResult } from '../index/types';
import type { Filter } from '../storage/types';
import type { GenerateOptions } from '../llm/types';

export interface RAGOptions {
  topK?: number;
  filter?: Filter;
  contextTemplate?: string;
  generateOptions?: GenerateOptions;
  maxContextTokens?: number;
  includeSourcesInResponse?: boolean;
}

export interface RAGResult {
  answer: string;
  sources: SearchResult[];
  metadata: {
    retrievalTime: number;
    generationTime: number;
    tokensGenerated?: number;
    contextLength?: number;
  };
}

export interface RAGStreamChunk {
  type: 'retrieval' | 'generation' | 'complete';
  content: string;
  sources?: SearchResult[];
  metadata?: {
    retrievalTime?: number;
    generationTime?: number;
  };
}

export interface RAGPipeline {
  query(query: string, options?: RAGOptions): Promise<RAGResult>;
  queryStream(query: string, options?: RAGOptions): AsyncGenerator<RAGStreamChunk>;
}
