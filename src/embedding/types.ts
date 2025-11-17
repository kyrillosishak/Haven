/**
 * Embedding layer types
 */

export interface EmbeddingGenerator {
  initialize(): Promise<void>;
  embed(text: string): Promise<Float32Array>;
  embedBatch(texts: string[]): Promise<Float32Array[]>;
  embedImage(image: ImageData | Blob): Promise<Float32Array>;
  getDimensions(): number;
  dispose(): Promise<void>;
}
