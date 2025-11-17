/**
 * Tests for WebLLM mock factory
 */

import { describe, it, expect } from 'vitest';
import { createMockMLCEngine } from './webllm.js';

describe('createMockMLCEngine', () => {
  it('should create a mock engine with correct structure', () => {
    const engine = createMockMLCEngine();
    
    expect(engine).toBeDefined();
    expect(engine.chat).toBeDefined();
    expect(engine.chat.completions).toBeDefined();
    expect(engine.chat.completions.create).toBeTypeOf('function');
    expect(engine.unload).toBeTypeOf('function');
    expect(engine.runtimeStatsText).toBeTypeOf('function');
    expect(engine.resetChat).toBeTypeOf('function');
  });

  it('should generate deterministic non-streaming responses', async () => {
    const engine = createMockMLCEngine();
    
    const completion = await engine.chat.completions.create({
      messages: [{ role: 'user', content: 'Hello' }],
    });
    
    expect(completion.choices).toHaveLength(1);
    expect(completion.choices[0].message?.content).toBeTruthy();
    expect(completion.choices[0].finish_reason).toBe('stop');
  });

  it('should generate same response for same prompt', async () => {
    const engine = createMockMLCEngine();
    
    const completion1 = await engine.chat.completions.create({
      messages: [{ role: 'user', content: 'Test prompt' }],
    });
    
    const completion2 = await engine.chat.completions.create({
      messages: [{ role: 'user', content: 'Test prompt' }],
    });
    
    expect(completion1.choices[0].message?.content).toBe(
      completion2.choices[0].message?.content
    );
  });

  it('should support streaming responses', async () => {
    const engine = createMockMLCEngine();
    
    const stream = engine.chat.completions.create({
      messages: [{ role: 'user', content: 'Hello' }],
      stream: true,
    });
    
    const chunks: string[] = [];
    for await (const chunk of stream as AsyncIterable<any>) {
      if (chunk.choices[0].delta?.content) {
        chunks.push(chunk.choices[0].delta.content);
      }
    }
    
    expect(chunks.length).toBeGreaterThan(0);
    const fullResponse = chunks.join('');
    expect(fullResponse).toBeTruthy();
  });

  it('should track mock state', async () => {
    const engine = createMockMLCEngine();
    
    expect(engine._mockState?.initialized).toBe(true);
    expect(engine._mockState?.disposed).toBe(false);
    expect(engine._mockState?.callCount).toBe(0);
    
    await engine.chat.completions.create({
      messages: [{ role: 'user', content: 'Test' }],
    });
    
    expect(engine._mockState?.callCount).toBe(1);
    expect(engine._mockState?.lastPrompt).toBe('Test');
  });

  it('should support custom responses', async () => {
    const customResponses = new Map([
      ['custom prompt', 'custom response'],
    ]);
    
    const engine = createMockMLCEngine({ responses: customResponses });
    
    const completion = await engine.chat.completions.create({
      messages: [{ role: 'user', content: 'custom prompt' }],
    });
    
    expect(completion.choices[0].message?.content).toBe('custom response');
  });

  it('should respect max_tokens parameter', async () => {
    const engine = createMockMLCEngine({
      defaultResponse: 'This is a very long response that should be truncated based on the max tokens parameter',
    });
    
    const completion = await engine.chat.completions.create({
      messages: [{ role: 'user', content: 'Test' }],
      max_tokens: 5,
    });
    
    const response = completion.choices[0].message?.content || '';
    const wordCount = response.split(' ').length;
    
    // Should be truncated (rough estimation)
    expect(wordCount).toBeLessThanOrEqual(10);
  });

  it('should handle unload correctly', async () => {
    const engine = createMockMLCEngine();
    
    await engine.chat.completions.create({
      messages: [{ role: 'user', content: 'Test' }],
    });
    
    expect(engine._mockState?.disposed).toBe(false);
    
    await engine.unload();
    
    expect(engine._mockState?.disposed).toBe(true);
    
    // Should throw after disposal
    expect(() => {
      engine.chat.completions.create({
        messages: [{ role: 'user', content: 'Test' }],
      });
    }).toThrow('Engine has been disposed');
  });

  it('should provide runtime stats', async () => {
    const engine = createMockMLCEngine();
    
    await engine.chat.completions.create({
      messages: [{ role: 'user', content: 'Test' }],
    });
    
    const stats = await engine.runtimeStatsText();
    
    expect(stats).toContain('Mock Runtime Stats');
    expect(stats).toContain('Calls: 1');
  });

  it('should reset chat history', async () => {
    const engine = createMockMLCEngine();
    
    await engine.chat.completions.create({
      messages: [{ role: 'user', content: 'Test' }],
    });
    
    expect(engine._mockState?.chatHistory.length).toBeGreaterThan(0);
    
    await engine.resetChat();
    
    expect(engine._mockState?.chatHistory.length).toBe(0);
    expect(engine._mockState?.lastPrompt).toBeNull();
  });

  it('should simulate delay when configured', async () => {
    const engine = createMockMLCEngine({ simulateDelay: 50 });
    
    const start = Date.now();
    await engine.chat.completions.create({
      messages: [{ role: 'user', content: 'Test' }],
    });
    const duration = Date.now() - start;
    
    expect(duration).toBeGreaterThanOrEqual(45); // Allow some margin
  });
});
