/**
 * WllamaProvider - WASM-based LLM inference using wllama
 */

import type { LLMProvider, GenerateOptions } from './types.js';

// Dynamic import type for wllama
type Wllama = any;

export interface WllamaProviderConfig {
  modelUrl: string;
  modelConfig?: {
    n_ctx?: number;
    n_batch?: number;
    n_threads?: number;
    embeddings?: boolean;
  };
  progressCallback?: (progress: { loaded: number; total: number }) => void;
  wasmPaths?: {
    'single-thread/wllama.wasm'?: string;
    'multi-thread/wllama.wasm'?: string;
    'multi-thread/wllama.worker.mjs'?: string;
  };
}

export class WllamaProvider implements LLMProvider {
  private wllama: Wllama | null = null;
  private config: WllamaProviderConfig;
  private initialized = false;
  private modelLoaded = false;

  constructor(config: WllamaProviderConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    try {
      // Dynamic import of wllama to avoid bundling issues
      const { Wllama } = await import('@wllama/wllama');

      // Initialize wllama instance
      this.wllama = new Wllama(this.config.wasmPaths || {} as any);

      this.initialized = true;

      // Load the model
      await this.loadModel();
    } catch (error) {
      throw new Error(
        `Failed to initialize WllamaProvider: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  private async loadModel(): Promise<void> {
    if (!this.wllama) {
      throw new Error('Wllama not initialized');
    }

    if (this.modelLoaded) {
      return;
    }

    try {
      // Load model with progress tracking
      await this.wllama.loadModelFromUrl(this.config.modelUrl, {
        n_ctx: this.config.modelConfig?.n_ctx || 2048,
        n_batch: this.config.modelConfig?.n_batch || 512,
        n_threads: this.config.modelConfig?.n_threads || 1,
        embeddings: this.config.modelConfig?.embeddings || false,
        progressCallback: this.config.progressCallback
          ? ({ loaded, total }: { loaded: number; total: number }) => {
              this.config.progressCallback?.({ loaded, total });
            }
          : undefined,
      });

      this.modelLoaded = true;
    } catch (error) {
      throw new Error(
        `Failed to load model from ${this.config.modelUrl}: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  async generate(prompt: string, options?: GenerateOptions): Promise<string> {
    if (!this.initialized || !this.wllama) {
      throw new Error('WllamaProvider not initialized. Call initialize() first.');
    }

    if (!this.modelLoaded) {
      throw new Error('Model not loaded');
    }

    try {
      const result = await this.wllama.createCompletion(prompt, {
        nPredict: options?.maxTokens || 512,
        sampling: {
          temp: options?.temperature ?? 0.7,
          top_p: options?.topP ?? 0.9,
          top_k: options?.topK ?? 40,
        },
        stopTokens: options?.stopSequences,
      });

      return result;
    } catch (error) {
      throw new Error(
        `Failed to generate text: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  async *generateStream(
    prompt: string,
    options?: GenerateOptions
  ): AsyncGenerator<string> {
    if (!this.initialized || !this.wllama) {
      throw new Error('WllamaProvider not initialized. Call initialize() first.');
    }

    if (!this.modelLoaded) {
      throw new Error('Model not loaded');
    }

    try {
      // Create a completion with streaming
      const stream = await this.wllama.createCompletion(prompt, {
        nPredict: options?.maxTokens || 512,
        sampling: {
          temp: options?.temperature ?? 0.7,
          top_p: options?.topP ?? 0.9,
          top_k: options?.topK ?? 40,
        },
        stopTokens: options?.stopSequences,
      });

      // If wllama supports streaming via async iteration
      if (Symbol.asyncIterator in stream) {
        for await (const chunk of stream as AsyncIterable<string>) {
          yield chunk;
        }
      } else {
        // Fallback: yield the complete result
        yield stream as string;
      }
    } catch (error) {
      throw new Error(
        `Failed to generate streaming text: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  async dispose(): Promise<void> {
    if (this.wllama) {
      try {
        // Exit and cleanup wllama resources
        await this.wllama.exit();
      } catch (error) {
        console.warn('Error during wllama cleanup:', error);
      }
      this.wllama = null;
      this.initialized = false;
      this.modelLoaded = false;
    }
  }

  /**
   * Check if the provider is initialized
   */
  isInitialized(): boolean {
    return this.initialized && this.modelLoaded;
  }

  /**
   * Get model information
   */
  getModelInfo(): { url: string; loaded: boolean } {
    return {
      url: this.config.modelUrl,
      loaded: this.modelLoaded,
    };
  }
}
