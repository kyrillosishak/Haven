/**
 * RAGPipelineManager - Orchestrates retrieval and generation for RAG workflows
 */

import type { VectorDB } from '../core/VectorDB';
import type { LLMProvider } from '../llm/types';
import type { EmbeddingGenerator } from '../embedding/types';
import type { SearchResult } from '../index/types';
import type { RAGPipeline, RAGOptions, RAGResult, RAGStreamChunk } from './types';
import { VectorDBError } from '../errors';

export interface RAGPipelineConfig {
  vectorDB: VectorDB;
  llmProvider: LLMProvider;
  embeddingGenerator: EmbeddingGenerator;
  defaultContextTemplate?: string;
  defaultMaxContextTokens?: number;
}

/**
 * RAGPipelineManager - Implements the RAG (Retrieval-Augmented Generation) pipeline
 * 
 * Orchestrates the complete RAG workflow:
 * 1. Embed the user query
 * 2. Search for relevant documents
 * 3. Format context from retrieved documents
 * 4. Build prompt with context injection
 * 5. Generate response using LLM
 */
export class RAGPipelineManager implements RAGPipeline {
  private vectorDB: VectorDB;
  private llmProvider: LLMProvider;
  private embeddingGenerator: EmbeddingGenerator;
  private defaultContextTemplate: string;
  private defaultMaxContextTokens: number;

  constructor(config: RAGPipelineConfig) {
    this.vectorDB = config.vectorDB;
    this.llmProvider = config.llmProvider;
    this.embeddingGenerator = config.embeddingGenerator;
    this.defaultContextTemplate = config.defaultContextTemplate || this.getDefaultTemplate();
    this.defaultMaxContextTokens = config.defaultMaxContextTokens || 2000;
  }

  /**
   * Execute a RAG query: retrieve relevant documents and generate a response
   * 
   * @param query - User query text
   * @param options - RAG options including topK, filters, and generation settings
   * @returns RAG result with answer, sources, and metadata
   */
  async query(query: string, options?: RAGOptions): Promise<RAGResult> {
    try {
      // Step 1: Retrieve relevant documents
      const retrievalStart = Date.now();
      const sources = await this.retrieve(query, options);
      const retrievalTime = Date.now() - retrievalStart;

      // Step 2: Format context from retrieved documents
      const context = this.formatContext(sources, options);

      // Step 3: Truncate context if needed
      const truncatedContext = this.truncateContext(
        context,
        options?.maxContextTokens || this.defaultMaxContextTokens
      );

      // Step 4: Build prompt with context injection
      const prompt = this.buildPrompt(query, truncatedContext);

      // Step 5: Generate response using LLM
      const generationStart = Date.now();
      const answer = await this.llmProvider.generate(prompt, options?.generateOptions);
      const generationTime = Date.now() - generationStart;

      // Step 6: Estimate token count
      const tokensGenerated = this.estimateTokenCount(answer);
      const contextLength = this.estimateTokenCount(truncatedContext);

      return {
        answer,
        sources: options?.includeSourcesInResponse !== false ? sources : [],
        metadata: {
          retrievalTime,
          generationTime,
          tokensGenerated,
          contextLength,
        },
      };
    } catch (error) {
      throw new VectorDBError(
        'Failed to execute RAG query',
        'RAG_QUERY_ERROR',
        { error, query }
      );
    }
  }

  /**
   * Execute a streaming RAG query: retrieve documents and stream the generated response
   * 
   * @param query - User query text
   * @param options - RAG options including topK, filters, and generation settings
   * @yields RAG stream chunks with retrieval results and generated text
   */
  async *queryStream(query: string, options?: RAGOptions): AsyncGenerator<RAGStreamChunk> {
    try {
      // Step 1: Retrieve relevant documents
      const retrievalStart = Date.now();
      const sources = await this.retrieve(query, options);
      const retrievalTime = Date.now() - retrievalStart;

      // Yield retrieval results
      yield {
        type: 'retrieval',
        content: '',
        sources: options?.includeSourcesInResponse !== false ? sources : [],
        metadata: { retrievalTime },
      };

      // Step 2: Format context from retrieved documents
      const context = this.formatContext(sources, options);

      // Step 3: Truncate context if needed
      const truncatedContext = this.truncateContext(
        context,
        options?.maxContextTokens || this.defaultMaxContextTokens
      );

      // Step 4: Build prompt with context injection
      const prompt = this.buildPrompt(query, truncatedContext);

      // Step 5: Stream generated response
      const generationStart = Date.now();
      let fullAnswer = '';

      for await (const chunk of this.llmProvider.generateStream(prompt, options?.generateOptions)) {
        fullAnswer += chunk;
        yield {
          type: 'generation',
          content: chunk,
        };
      }

      const generationTime = Date.now() - generationStart;

      // Yield completion metadata
      yield {
        type: 'complete',
        content: '',
        metadata: {
          retrievalTime,
          generationTime,
        },
      };
    } catch (error) {
      throw new VectorDBError(
        'Failed to execute streaming RAG query',
        'RAG_STREAM_ERROR',
        { error, query }
      );
    }
  }

  /**
   * Retrieve relevant documents for a query
   * 
   * @param query - User query text
   * @param options - RAG options with topK and filter
   * @returns Array of search results
   */
  private async retrieve(query: string, options?: RAGOptions): Promise<SearchResult[]> {
    // Generate query embedding
    const queryVector = await this.embeddingGenerator.embed(query);

    // Search for relevant documents
    const results = await this.vectorDB.search({
      vector: queryVector,
      k: options?.topK || 5,
      filter: options?.filter,
      includeVectors: false,
    });

    return results;
  }

  /**
   * Format context from retrieved documents using a template
   * 
   * @param results - Search results to format
   * @param options - RAG options with optional context template
   * @returns Formatted context string
   */
  private formatContext(results: SearchResult[], options?: RAGOptions): string {
    if (results.length === 0) {
      return 'No relevant information found.';
    }

    const template = options?.contextTemplate || this.defaultContextTemplate;

    // Format each result using the template
    const formattedResults = results.map((result, index) => {
      return this.applyTemplate(template, result, index);
    });

    return formattedResults.join('\n\n');
  }

  /**
   * Apply a template to a search result
   * 
   * @param template - Template string with placeholders
   * @param result - Search result to format
   * @param index - Result index (0-based)
   * @returns Formatted string
   */
  private applyTemplate(template: string, result: SearchResult, index: number): string {
    let formatted = template;

    // Replace placeholders
    formatted = formatted.replace(/\{index\}/g, String(index + 1));
    formatted = formatted.replace(/\{score\}/g, result.score.toFixed(4));
    formatted = formatted.replace(/\{content\}/g, result.metadata.content || '');
    formatted = formatted.replace(/\{title\}/g, result.metadata.title || '');
    formatted = formatted.replace(/\{url\}/g, result.metadata.url || '');
    formatted = formatted.replace(/\{id\}/g, result.id);

    // Replace any custom metadata fields
    formatted = formatted.replace(/\{metadata\.(\w+)\}/g, (_match, field) => {
      return result.metadata[field] !== undefined ? String(result.metadata[field]) : '';
    });

    return formatted;
  }

  /**
   * Build a prompt with context injection
   * 
   * @param query - User query
   * @param context - Formatted context from retrieved documents
   * @returns Complete prompt for LLM
   */
  private buildPrompt(query: string, context: string): string {
    return `You are a helpful assistant. Use the following context to answer the user's question. If the context doesn't contain relevant information, say so.

Context:
${context}

Question: ${query}

Answer:`;
  }

  /**
   * Truncate context to fit within token limit
   * 
   * @param context - Context string to truncate
   * @param maxTokens - Maximum number of tokens
   * @returns Truncated context
   */
  private truncateContext(context: string, maxTokens: number): string {
    // Rough estimation: 1 token ≈ 4 characters
    const maxChars = maxTokens * 4;

    if (context.length <= maxChars) {
      return context;
    }

    // Truncate and add ellipsis
    const truncated = context.substring(0, maxChars);
    
    // Try to truncate at a sentence boundary
    const lastPeriod = truncated.lastIndexOf('.');
    const lastNewline = truncated.lastIndexOf('\n');
    const cutoff = Math.max(lastPeriod, lastNewline);

    if (cutoff > maxChars * 0.8) {
      // If we found a good boundary, use it
      return truncated.substring(0, cutoff + 1) + '\n\n[Context truncated due to length...]';
    }

    // Otherwise, just truncate at character limit
    return truncated + '...\n\n[Context truncated due to length...]';
  }

  /**
   * Estimate token count for a text string
   * 
   * @param text - Text to estimate
   * @returns Estimated token count
   */
  private estimateTokenCount(text: string): number {
    // Rough estimation: 1 token ≈ 4 characters
    // This is a simple heuristic; actual tokenization varies by model
    return Math.ceil(text.length / 4);
  }

  /**
   * Get the default context template
   * 
   * @returns Default template string
   */
  private getDefaultTemplate(): string {
    return `Document {index}:
{content}`;
  }

  /**
   * Set a custom context template
   * 
   * @param template - Template string with placeholders
   */
  setContextTemplate(template: string): void {
    this.defaultContextTemplate = template;
  }

  /**
   * Set the default maximum context tokens
   * 
   * @param maxTokens - Maximum number of tokens for context
   */
  setMaxContextTokens(maxTokens: number): void {
    this.defaultMaxContextTokens = maxTokens;
  }

  /**
   * Get current configuration
   * 
   * @returns Current RAG pipeline configuration
   */
  getConfig(): {
    defaultContextTemplate: string;
    defaultMaxContextTokens: number;
  } {
    return {
      defaultContextTemplate: this.defaultContextTemplate,
      defaultMaxContextTokens: this.defaultMaxContextTokens,
    };
  }
}
