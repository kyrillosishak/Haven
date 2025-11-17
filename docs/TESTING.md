# Testing Guide

## Overview

This guide explains the testing strategy for browser-vector-db, including when to use unit tests vs integration tests, how to write tests with mocks, and how to run different test suites.

## Testing Philosophy

Our testing approach follows these principles:

1. **Test your code, not external dependencies** - We mock ML models (Transformers.js, WebLLM) because we're testing our wrapper code, not whether HuggingFace models work
2. **Fast feedback loop** - Unit tests run in seconds without network access
3. **Test behavior, not implementation** - Focus on what your code does, not how it does it
4. **Confidence over coverage** - Write tests that give you confidence your code works

### What Should Be Tested?

✅ **DO test:**
- Your code's behavior (initialization, error handling, state management)
- API contracts (correct parameters passed to external libraries)
- Error handling (graceful failures, retries, fallbacks)
- State management (initialization/disposal lifecycle)
- User-facing functionality (can users successfully use your API?)

❌ **DON'T test:**
- Whether HuggingFace models produce correct embeddings (that's their responsibility)
- Whether WebLLM correctly generates text (that's their responsibility)
- Implementation details of external libraries

## Test Types

### Unit Tests (Mocked)

**Purpose**: Test YOUR wrapper code behavior without external dependencies

**Characteristics**:
- Use mocked ML libraries (Transformers.js, WebLLM)
- No network access required
- Fast execution (< 30 seconds for full suite)
- Run on every commit

**File naming**: `*.test.ts`

**Example**: `src/embedding/TransformersEmbedding.test.ts`

### Integration Tests (Real Models)

**Purpose**: Verify real model integration in actual browser environments

**Characteristics**:
- Use real Transformers.js and WebLLM
- Require internet connection for model downloads
- Slow execution (several minutes)
- Run before releases, not on every commit
- Skipped in CI by default

**File naming**: `*.integration.test.ts`

**Example**: `src/embedding/TransformersEmbedding.integration.test.ts`

## Running Tests

### Run All Unit Tests (Default)

```bash
npm test
```

This runs all unit tests with mocked dependencies. Fast and suitable for development.

### Run Tests in Watch Mode

```bash
npm run test:watch
```

Automatically re-runs tests when files change. Excludes integration tests.

### Run Integration Tests Only

```bash
npm run test:integration
```

Runs only integration tests with real models. Requires network access.

### Run Tests with Coverage

```bash
npm run test:coverage
```

Generates code coverage report. Excludes integration tests and mock files.

### Skip Integration Tests

```bash
SKIP_INTEGRATION=true npm test
```

Explicitly skip integration tests (useful in CI environments).

## Writing Unit Tests with Mocks

### Basic Test Structure

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TransformersEmbedding } from './TransformersEmbedding';
import { createMockPipeline } from '../test/mocks';

// Mock the entire module
vi.mock('@huggingface/transformers', () => ({
  pipeline: vi.fn(),
  env: {
    allowLocalModels: true,
    useBrowserCache: false,
  },
}));

describe('TransformersEmbedding', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should initialize with correct model and options', async () => {
    const { pipeline } = await import('@huggingface/transformers');
    vi.mocked(pipeline).mockResolvedValue(createMockPipeline());

    const embedding = new TransformersEmbedding({ 
      model: 'test-model',
      device: 'webgpu',
    });
    
    await embedding.initialize();
    
    // Test YOUR code's behavior
    expect(pipeline).toHaveBeenCalledWith(
      'feature-extraction',
      'test-model',
      expect.objectContaining({ device: 'webgpu' })
    );
    expect(embedding.getDimensions()).toBeGreaterThan(0);
  });
});
```

### Testing Error Handling

```typescript
it('should handle initialization failures with retry', async () => {
  const { pipeline } = await import('@huggingface/transformers');
  const mockPipeline = vi.fn()
    .mockRejectedValueOnce(new Error('Network error'))
    .mockRejectedValueOnce(new Error('Network error'))
    .mockResolvedValueOnce(createMockPipeline());
  
  vi.mocked(pipeline).mockImplementation(mockPipeline);
  
  const embedding = new TransformersEmbedding({ 
    model: 'test-model',
    maxRetries: 3,
  });
  
  await embedding.initialize();
  
  // Verify retry logic works
  expect(mockPipeline).toHaveBeenCalledTimes(3);
  expect(embedding.getDimensions()).toBeGreaterThan(0);
});
```

### Testing State Management

```typescript
it('should track initialization state correctly', async () => {
  const { pipeline } = await import('@huggingface/transformers');
  vi.mocked(pipeline).mockResolvedValue(createMockPipeline());

  const embedding = new TransformersEmbedding({ model: 'test-model' });
  
  expect(embedding.getDimensions()).toBe(0); // Not initialized
  
  await embedding.initialize();
  
  expect(embedding.getDimensions()).toBeGreaterThan(0); // Initialized
  
  await embedding.dispose();
  
  expect(embedding.getDimensions()).toBe(0); // Disposed
});
```

## Mock API Reference

### Transformers.js Mocks

#### `createMockPipeline(options?)`

Creates a mock Transformers.js pipeline for testing.

**Options**:
```typescript
interface MockPipelineOptions {
  dimensions?: number;              // Embedding dimensions (default: 384)
  deterministicEmbeddings?: boolean; // Generate consistent embeddings (default: true)
  simulateDelay?: number;           // Simulate processing delay in ms (default: 0)
}
```

**Usage**:
```typescript
import { createMockPipeline } from '../test/mocks';

const mockPipeline = createMockPipeline({
  dimensions: 512,
  simulateDelay: 10,
});

// Use in tests
const result = await mockPipeline('test text');
expect(result.data).toHaveLength(512);
```

**Behavior**:
- Generates deterministic embeddings based on text hash
- Returns normalized Float32Array vectors
- Supports `dispose()` method
- Tracks call count and state

### WebLLM Mocks

#### `createMockMLCEngine(options?)`

Creates a mock WebLLM MLCEngine for testing.

**Options**:
```typescript
interface MockMLCEngineOptions {
  simulateDelay?: number;           // Delay per chunk in ms (default: 0)
  responses?: Map<string, string>;  // Custom prompt → response mapping
  defaultResponse?: string;         // Default response text
}
```

**Usage**:
```typescript
import { createMockMLCEngine } from '../test/mocks';

const mockEngine = createMockMLCEngine({
  defaultResponse: 'This is a test response',
  simulateDelay: 5,
});

// Non-streaming generation
const response = await mockEngine.chat.completions.create({
  messages: [{ role: 'user', content: 'Hello' }],
});

// Streaming generation
const stream = await mockEngine.chat.completions.create({
  messages: [{ role: 'user', content: 'Hello' }],
  stream: true,
});

for await (const chunk of stream) {
  console.log(chunk.choices[0].delta.content);
}
```

**Behavior**:
- Returns deterministic responses based on prompts
- Supports both streaming and non-streaming
- Respects `max_tokens` and other generation options
- Implements `unload()`, `resetChat()`, and `runtimeStatsText()`

### Utility Functions

#### `setupTransformersMocks()`

Sets up all Transformers.js mocks with default configuration.

```typescript
import { setupTransformersMocks } from '../test/mocks';

beforeEach(() => {
  setupTransformersMocks();
});
```

#### `setupWebLLMMocks()`

Sets up all WebLLM mocks with default configuration.

```typescript
import { setupWebLLMMocks } from '../test/mocks';

beforeEach(() => {
  setupWebLLMMocks();
});
```

#### `setupAllMocks()`

Sets up all mocks (Transformers.js + WebLLM).

```typescript
import { setupAllMocks } from '../test/mocks';

beforeEach(() => {
  setupAllMocks();
});
```

#### `resetAllMocks()`

Resets all mock state and call counts.

```typescript
import { resetAllMocks } from '../test/mocks';

afterEach(() => {
  resetAllMocks();
});
```

## Common Testing Patterns

### Pattern 1: Testing Initialization

```typescript
it('should initialize with correct parameters', async () => {
  const { pipeline } = await import('@huggingface/transformers');
  vi.mocked(pipeline).mockResolvedValue(createMockPipeline());

  const embedding = new TransformersEmbedding({
    model: 'Xenova/all-MiniLM-L6-v2',
    device: 'webgpu',
    quantized: true,
  });

  await embedding.initialize();

  expect(pipeline).toHaveBeenCalledWith(
    'feature-extraction',
    'Xenova/all-MiniLM-L6-v2',
    expect.objectContaining({
      device: 'webgpu',
      quantized: true,
    })
  );
});
```

### Pattern 2: Testing Error Recovery

```typescript
it('should fall back to WASM when WebGPU fails', async () => {
  const { pipeline } = await import('@huggingface/transformers');
  
  // First call fails with WebGPU
  vi.mocked(pipeline)
    .mockRejectedValueOnce(new Error('WebGPU not supported'))
    .mockResolvedValueOnce(createMockPipeline());

  const embedding = new TransformersEmbedding({
    model: 'test-model',
    device: 'webgpu',
    fallbackToWasm: true,
  });

  await embedding.initialize();

  // Should have tried twice: WebGPU then WASM
  expect(pipeline).toHaveBeenCalledTimes(2);
  expect(pipeline).toHaveBeenLastCalledWith(
    'feature-extraction',
    'test-model',
    expect.objectContaining({ device: 'wasm' })
  );
});
```

### Pattern 3: Testing Async Operations

```typescript
it('should generate embeddings for text', async () => {
  const { pipeline } = await import('@huggingface/transformers');
  const mockPipeline = createMockPipeline({ dimensions: 384 });
  vi.mocked(pipeline).mockResolvedValue(mockPipeline);

  const embedding = new TransformersEmbedding({ model: 'test-model' });
  await embedding.initialize();

  const result = await embedding.embed('test text');

  expect(result).toBeInstanceOf(Float32Array);
  expect(result).toHaveLength(384);
});
```

### Pattern 4: Testing Streaming

```typescript
it('should stream text generation', async () => {
  const { CreateMLCEngine } = await import('@mlc-ai/web-llm');
  const mockEngine = createMockMLCEngine({
    defaultResponse: 'Hello world',
  });
  vi.mocked(CreateMLCEngine).mockResolvedValue(mockEngine);

  const provider = new WebLLMProvider({ model: 'test-model' });
  await provider.initialize();

  const chunks: string[] = [];
  const stream = await provider.generate('test prompt', { stream: true });

  for await (const chunk of stream) {
    if (chunk.choices[0].delta.content) {
      chunks.push(chunk.choices[0].delta.content);
    }
  }

  expect(chunks.length).toBeGreaterThan(0);
  expect(chunks.join('')).toContain('Hello');
});
```

### Pattern 5: Testing Batch Operations

```typescript
it('should embed multiple texts in batch', async () => {
  const { pipeline } = await import('@huggingface/transformers');
  const mockPipeline = createMockPipeline({ dimensions: 384 });
  vi.mocked(pipeline).mockResolvedValue(mockPipeline);

  const embedding = new TransformersEmbedding({ model: 'test-model' });
  await embedding.initialize();

  const texts = ['text 1', 'text 2', 'text 3'];
  const results = await embedding.embedBatch(texts);

  expect(results).toHaveLength(3);
  results.forEach(result => {
    expect(result).toBeInstanceOf(Float32Array);
    expect(result).toHaveLength(384);
  });
});
```

### Pattern 6: Testing Disposal and Cleanup

```typescript
it('should clean up resources on disposal', async () => {
  const { pipeline } = await import('@huggingface/transformers');
  const mockPipeline = createMockPipeline();
  const disposeSpy = vi.spyOn(mockPipeline, 'dispose');
  vi.mocked(pipeline).mockResolvedValue(mockPipeline);

  const embedding = new TransformersEmbedding({ model: 'test-model' });
  await embedding.initialize();
  await embedding.dispose();

  expect(disposeSpy).toHaveBeenCalled();
  expect(embedding.getDimensions()).toBe(0);
});
```

## Writing Integration Tests

Integration tests verify real model integration in actual browser environments. These should be used sparingly and run before releases.

### Basic Integration Test

```typescript
import { describe, it, expect } from 'vitest';
import { TransformersEmbedding } from './TransformersEmbedding';

const skipIntegration = process.env.CI || process.env.SKIP_INTEGRATION;

describe('TransformersEmbedding (Integration)', () => {
  it.skipIf(skipIntegration)('should load real model and generate embeddings', async () => {
    const embedding = new TransformersEmbedding({
      model: 'Xenova/all-MiniLM-L6-v2',
      device: 'wasm', // Use WASM for CI compatibility
    });

    await embedding.initialize();

    const result = await embedding.embed('test text');

    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(384);
    
    // Verify embedding is normalized
    const norm = Math.sqrt(
      Array.from(result).reduce((sum, val) => sum + val * val, 0)
    );
    expect(norm).toBeCloseTo(1.0, 5);

    await embedding.dispose();
  }, 60000); // 60 second timeout for model download
});
```

### When to Write Integration Tests

Write integration tests when:
- Testing end-to-end user workflows
- Verifying browser compatibility (WebGPU, WASM)
- Validating model loading and initialization
- Testing before major releases

Don't write integration tests for:
- Every feature or bug fix
- Testing your wrapper code logic
- Testing error handling (use mocks instead)
- Rapid development iterations

## Best Practices

### 1. Use Descriptive Test Names

```typescript
// ✅ Good
it('should retry initialization 3 times before failing', async () => {});

// ❌ Bad
it('should work', async () => {});
```

### 2. Test One Thing Per Test

```typescript
// ✅ Good
it('should initialize with correct model', async () => {});
it('should handle initialization errors', async () => {});

// ❌ Bad
it('should initialize and handle errors and dispose', async () => {});
```

### 3. Use beforeEach for Setup

```typescript
describe('TransformersEmbedding', () => {
  let embedding: TransformersEmbedding;

  beforeEach(async () => {
    const { pipeline } = await import('@huggingface/transformers');
    vi.mocked(pipeline).mockResolvedValue(createMockPipeline());
    
    embedding = new TransformersEmbedding({ model: 'test-model' });
    await embedding.initialize();
  });

  afterEach(async () => {
    await embedding.dispose();
  });

  it('should generate embeddings', async () => {
    const result = await embedding.embed('test');
    expect(result).toBeInstanceOf(Float32Array);
  });
});
```

### 4. Clear Mocks Between Tests

```typescript
beforeEach(() => {
  vi.clearAllMocks();
});
```

### 5. Test Error Messages

```typescript
it('should throw descriptive error when model fails to load', async () => {
  const { pipeline } = await import('@huggingface/transformers');
  vi.mocked(pipeline).mockRejectedValue(new Error('Model not found'));

  const embedding = new TransformersEmbedding({ model: 'invalid-model' });

  await expect(embedding.initialize()).rejects.toThrow('Model not found');
});
```

### 6. Use Type-Safe Mocks

```typescript
import { vi } from 'vitest';
import type { Pipeline } from '@huggingface/transformers';

const mockPipeline = createMockPipeline() as unknown as Pipeline;
```

## Troubleshooting

### Tests Fail with "Cannot find module"

Make sure you're using `vi.mock()` at the top level of your test file:

```typescript
// ✅ Correct - at top level
vi.mock('@huggingface/transformers', () => ({
  pipeline: vi.fn(),
}));

describe('MyTest', () => {
  // ❌ Wrong - inside describe
  vi.mock('@huggingface/transformers');
});
```

### Mocks Not Working

Ensure you're clearing mocks between tests:

```typescript
beforeEach(() => {
  vi.clearAllMocks();
});
```

### Integration Tests Running in CI

Set the `SKIP_INTEGRATION` environment variable:

```bash
SKIP_INTEGRATION=true npm test
```

Or use the skip condition in your test:

```typescript
const skipIntegration = process.env.CI || process.env.SKIP_INTEGRATION;

it.skipIf(skipIntegration)('integration test', async () => {});
```

### Async Tests Timing Out

Increase the timeout for slow operations:

```typescript
it('should load large model', async () => {
  // test code
}, 60000); // 60 second timeout
```

## Summary

- **Unit tests** (mocked): Fast, run on every commit, test your code's behavior
- **Integration tests** (real): Slow, run before releases, test end-to-end functionality
- **Mock external dependencies**: Models are external dependencies like APIs
- **Test behavior, not implementation**: Focus on what your code does
- **Fast feedback loop**: Unit tests should run in seconds

For more information, see the [Design Document](./.kiro/specs/test-mocking-strategy/design.md).
