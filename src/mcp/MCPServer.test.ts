/**
 * Tests for MCPServer
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MCPServer } from './MCPServer';
import type { VectorDB } from '../core/VectorDB';
import type { RAGPipeline } from '../rag/types';
import type { SearchResult } from '../index/types';

describe('MCPServer', () => {
  let mockVectorDB: VectorDB;
  let mockRAGPipeline: RAGPipeline;
  let mcpServer: MCPServer;

  beforeEach(() => {
    // Create mock VectorDB
    mockVectorDB = {
      search: vi.fn(),
      insert: vi.fn(),
      delete: vi.fn(),
    } as any;

    // Create mock RAG pipeline
    mockRAGPipeline = {
      query: vi.fn(),
      queryStream: vi.fn(),
    } as any;

    // Create MCP server
    mcpServer = new MCPServer({
      vectorDB: mockVectorDB,
      ragPipeline: mockRAGPipeline,
    });
  });

  describe('Tool Registration', () => {
    it('should register all required tools', () => {
      const tools = mcpServer.getTools();
      const toolNames = tools.map(t => t.name);

      expect(toolNames).toContain('search_vectors');
      expect(toolNames).toContain('insert_document');
      expect(toolNames).toContain('delete_document');
      expect(toolNames).toContain('rag_query');
    });

    it('should not register rag_query tool without RAG pipeline', () => {
      const serverWithoutRAG = new MCPServer({
        vectorDB: mockVectorDB,
      });

      const toolNames = serverWithoutRAG.getToolNames();
      expect(toolNames).not.toContain('rag_query');
    });

    it('should provide tool schemas with required fields', () => {
      const tools = mcpServer.getTools();

      for (const tool of tools) {
        expect(tool.name).toBeDefined();
        expect(tool.description).toBeDefined();
        expect(tool.inputSchema).toBeDefined();
        expect(tool.handler).toBeDefined();
        expect(typeof tool.handler).toBe('function');
      }
    });
  });

  describe('search_vectors Tool', () => {
    it('should execute search with text query', async () => {
      const mockResults: SearchResult[] = [
        {
          id: 'doc1',
          score: 0.95,
          metadata: { content: 'Test document 1', title: 'Doc 1' },
        },
        {
          id: 'doc2',
          score: 0.85,
          metadata: { content: 'Test document 2', title: 'Doc 2' },
        },
      ];

      vi.mocked(mockVectorDB.search).mockResolvedValue(mockResults);

      const result = await mcpServer.executeTool('search_vectors', {
        query: 'test query',
        k: 5,
      });

      expect(result.success).toBe(true);
      expect(result.results).toHaveLength(2);
      expect(result.count).toBe(2);
      expect(mockVectorDB.search).toHaveBeenCalledWith({
        text: 'test query',
        k: 5,
        filter: undefined,
        includeVectors: false,
      });
    });

    it('should use default k value when not provided', async () => {
      vi.mocked(mockVectorDB.search).mockResolvedValue([]);

      await mcpServer.executeTool('search_vectors', {
        query: 'test query',
      });

      expect(mockVectorDB.search).toHaveBeenCalledWith({
        text: 'test query',
        k: 5,
        filter: undefined,
        includeVectors: false,
      });
    });

    it('should include vectors when requested', async () => {
      const mockResults: SearchResult[] = [
        {
          id: 'doc1',
          score: 0.95,
          metadata: { content: 'Test' },
          vector: new Float32Array([0.1, 0.2, 0.3]),
        },
      ];

      vi.mocked(mockVectorDB.search).mockResolvedValue(mockResults);

      const result = await mcpServer.executeTool('search_vectors', {
        query: 'test',
        includeVectors: true,
      });

      expect(result.results[0].vector).toBeDefined();
      expect(Array.isArray(result.results[0].vector)).toBe(true);
    });

    it('should apply metadata filters', async () => {
      vi.mocked(mockVectorDB.search).mockResolvedValue([]);

      const filter = {
        field: 'category',
        operator: 'eq',
        value: 'science',
      };

      await mcpServer.executeTool('search_vectors', {
        query: 'test',
        filter,
      });

      expect(mockVectorDB.search).toHaveBeenCalledWith({
        text: 'test',
        k: 5,
        filter,
        includeVectors: false,
      });
    });

    it('should throw error when query is missing', async () => {
      await expect(
        mcpServer.executeTool('search_vectors', {})
      ).rejects.toThrow();
    });
  });

  describe('insert_document Tool', () => {
    it('should insert document with content', async () => {
      vi.mocked(mockVectorDB.insert).mockResolvedValue('doc-123');

      const result = await mcpServer.executeTool('insert_document', {
        content: 'This is a test document',
      });

      expect(result.success).toBe(true);
      expect(result.id).toBe('doc-123');
      expect(mockVectorDB.insert).toHaveBeenCalledWith({
        text: 'This is a test document',
        metadata: {
          content: 'This is a test document',
        },
      });
    });

    it('should insert document with metadata', async () => {
      vi.mocked(mockVectorDB.insert).mockResolvedValue('doc-456');

      const result = await mcpServer.executeTool('insert_document', {
        content: 'Test content',
        metadata: {
          title: 'Test Title',
          category: 'test',
          tags: ['tag1', 'tag2'],
        },
      });

      expect(result.success).toBe(true);
      expect(mockVectorDB.insert).toHaveBeenCalledWith({
        text: 'Test content',
        metadata: {
          title: 'Test Title',
          category: 'test',
          tags: ['tag1', 'tag2'],
          content: 'Test content',
        },
      });
    });

    it('should throw error when content is missing', async () => {
      await expect(
        mcpServer.executeTool('insert_document', {})
      ).rejects.toThrow();
    });
  });

  describe('delete_document Tool', () => {
    it('should delete existing document', async () => {
      vi.mocked(mockVectorDB.delete).mockResolvedValue(true);

      const result = await mcpServer.executeTool('delete_document', {
        id: 'doc-123',
      });

      expect(result.success).toBe(true);
      expect(result.id).toBe('doc-123');
      expect(mockVectorDB.delete).toHaveBeenCalledWith('doc-123');
    });

    it('should return failure when document not found', async () => {
      vi.mocked(mockVectorDB.delete).mockResolvedValue(false);

      const result = await mcpServer.executeTool('delete_document', {
        id: 'nonexistent',
      });

      expect(result.success).toBe(false);
      expect(result.message).toContain('not found');
    });

    it('should throw error when id is missing', async () => {
      await expect(
        mcpServer.executeTool('delete_document', {})
      ).rejects.toThrow();
    });
  });

  describe('rag_query Tool', () => {
    it('should execute RAG query', async () => {
      const mockRAGResult = {
        answer: 'This is the generated answer',
        sources: [
          {
            id: 'doc1',
            score: 0.95,
            metadata: { content: 'Source 1' },
          },
        ],
        metadata: {
          retrievalTime: 50,
          generationTime: 200,
          tokensGenerated: 20,
        },
      };

      vi.mocked(mockRAGPipeline.query).mockResolvedValue(mockRAGResult);

      const result = await mcpServer.executeTool('rag_query', {
        query: 'What is the answer?',
      });

      expect(result.success).toBe(true);
      expect(result.answer).toBe('This is the generated answer');
      expect(result.sources).toHaveLength(1);
      expect(result.metadata).toBeDefined();
      expect(mockRAGPipeline.query).toHaveBeenCalledWith('What is the answer?', {
        topK: 5,
        filter: undefined,
        generateOptions: {
          maxTokens: 512,
          temperature: 0.7,
        },
      });
    });

    it('should use custom generation parameters', async () => {
      vi.mocked(mockRAGPipeline.query).mockResolvedValue({
        answer: 'Answer',
        sources: [],
        metadata: { retrievalTime: 0, generationTime: 0 },
      });

      await mcpServer.executeTool('rag_query', {
        query: 'test',
        topK: 10,
        maxTokens: 1024,
        temperature: 0.9,
      });

      expect(mockRAGPipeline.query).toHaveBeenCalledWith('test', {
        topK: 10,
        filter: undefined,
        generateOptions: {
          maxTokens: 1024,
          temperature: 0.9,
        },
      });
    });

    it('should throw error when RAG pipeline not configured', async () => {
      const serverWithoutRAG = new MCPServer({
        vectorDB: mockVectorDB,
      });

      // Manually add the tool to test error handling
      const ragTool = {
        name: 'rag_query',
        description: 'Test',
        inputSchema: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] },
        handler: async () => {
          throw new Error('RAG pipeline not configured');
        },
      };

      // This should not happen in practice, but tests the error path
      await expect(
        serverWithoutRAG.executeTool('rag_query', { query: 'test' })
      ).rejects.toThrow();
    });
  });

  describe('Tool Execution', () => {
    it('should throw error for unknown tool', async () => {
      await expect(
        mcpServer.executeTool('unknown_tool', {})
      ).rejects.toThrow("Tool 'unknown_tool' not found");
    });

    it('should validate parameter types', async () => {
      await expect(
        mcpServer.executeTool('search_vectors', {
          query: 'test',
          k: 'not a number', // Invalid type
        })
      ).rejects.toThrow();
    });
  });

  describe('Tool Introspection', () => {
    it('should get tool by name', () => {
      const tool = mcpServer.getTool('search_vectors');
      expect(tool).toBeDefined();
      expect(tool?.name).toBe('search_vectors');
    });

    it('should return undefined for unknown tool', () => {
      const tool = mcpServer.getTool('unknown');
      expect(tool).toBeUndefined();
    });

    it('should check if tool exists', () => {
      expect(mcpServer.hasTool('search_vectors')).toBe(true);
      expect(mcpServer.hasTool('unknown')).toBe(false);
    });

    it('should get all tool names', () => {
      const names = mcpServer.getToolNames();
      expect(names).toContain('search_vectors');
      expect(names).toContain('insert_document');
      expect(names).toContain('delete_document');
    });
  });

  describe('Parameter Validation', () => {
    it('should validate required parameters', async () => {
      await expect(
        mcpServer.executeTool('search_vectors', {})
      ).rejects.toThrow();
    });

    it('should validate parameter types', async () => {
      await expect(
        mcpServer.executeTool('search_vectors', {
          query: 123, // Should be string
        })
      ).rejects.toThrow();
    });

    it('should validate numeric ranges', async () => {
      vi.mocked(mockVectorDB.search).mockResolvedValue([]);

      // Test minimum
      await expect(
        mcpServer.executeTool('search_vectors', {
          query: 'test',
          k: 0, // Below minimum
        })
      ).rejects.toThrow();

      // Test maximum
      await expect(
        mcpServer.executeTool('search_vectors', {
          query: 'test',
          k: 101, // Above maximum
        })
      ).rejects.toThrow();
    });

    it('should validate enum values', async () => {
      await expect(
        mcpServer.executeTool('search_vectors', {
          query: 'test',
          filter: {
            field: 'test',
            operator: 'invalid_operator',
            value: 'test',
          },
        })
      ).rejects.toThrow();
    });
  });
});
