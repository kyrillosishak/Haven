import { describe, it, expect } from 'vitest';
import { createMockPipeline } from './transformers';

describe('createMockPipeline', () => {
  it('should generate embeddings with correct dimensions', async () => {
    const pipeline = createMockPipeline({ dimensions: 384 });
    const result = await pipeline('test text');
    
    expect(result.data).toBeInstanceOf(Float32Array);
    expect(result.data.length).toBe(384);
  });

  it('should generate deterministic embeddings for same text', async () => {
    const pipeline = createMockPipeline({ dimensions: 384 });
    const result1 = await pipeline('test text');
    const result2 = await pipeline('test text');
    
    expect(Array.from(result1.data)).toEqual(Array.from(result2.data));
  });

  it('should generate different embeddings for different text', async () => {
    const pipeline = createMockPipeline({ dimensions: 384 });
    const result1 = await pipeline('test text');
    const result2 = await pipeline('different text');
    
    expect(Array.from(result1.data)).not.toEqual(Array.from(result2.data));
  });

  it('should normalize embeddings to unit vectors', async () => {
    const pipeline = createMockPipeline({ dimensions: 384 });
    const result = await pipeline('test text');
    
    const norm = Math.sqrt(
      result.data.reduce((sum, val) => sum + val * val, 0)
    );
    
    expect(norm).toBeCloseTo(1.0, 5);
  });

  it('should track call count', async () => {
    const pipeline = createMockPipeline({ dimensions: 384 });
    
    expect(pipeline._mockState?.callCount).toBe(0);
    
    await pipeline('text 1');
    expect(pipeline._mockState?.callCount).toBe(1);
    
    await pipeline('text 2');
    expect(pipeline._mockState?.callCount).toBe(2);
  });

  it('should track last input and output', async () => {
    const pipeline = createMockPipeline({ dimensions: 384 });
    const result = await pipeline('test text');
    
    expect(pipeline._mockState?.lastInput).toBe('test text');
    expect(pipeline._mockState?.lastOutput).toBe(result);
  });

  it('should support disposal', async () => {
    const pipeline = createMockPipeline({ dimensions: 384 });
    
    expect(pipeline._mockState?.disposed).toBe(false);
    
    await pipeline.dispose?.();
    
    expect(pipeline._mockState?.disposed).toBe(true);
  });

  it('should throw error when used after disposal', async () => {
    const pipeline = createMockPipeline({ dimensions: 384 });
    await pipeline.dispose?.();
    
    await expect(pipeline('test text')).rejects.toThrow('Pipeline has been disposed');
  });

  it('should support tolist() method on output', async () => {
    const pipeline = createMockPipeline({ dimensions: 384 });
    const result = await pipeline('test text');
    
    const list = result.tolist();
    expect(Array.isArray(list)).toBe(true);
    expect(list.length).toBe(384);
    expect(list).toEqual(Array.from(result.data));
  });

  it('should support custom dimensions', async () => {
    const pipeline = createMockPipeline({ dimensions: 768 });
    const result = await pipeline('test text');
    
    expect(result.data.length).toBe(768);
  });
});
