# Performance Benchmarking Guide

This guide explains how to run and interpret performance benchmarks for the VectorDB system.

## Overview

The benchmarking suite provides comprehensive performance testing including:

- **Search Latency**: Measure search performance across different dataset sizes
- **Insertion Throughput**: Measure how many documents can be inserted per second
- **Memory Usage**: Profile memory consumption during operations
- **Model Load Times**: Measure embedding model initialization time
- **Cache Performance**: Evaluate cache hit rates and performance impact

## Quick Start

### Running Benchmarks Programmatically

Create a file `my-benchmark.js`:

```javascript
import { BenchmarkRunner } from '@vectordb/browser-vectordb';

async function main() {
  const runner = new BenchmarkRunner({
    datasetSizes: [100, 1000, 10000],
    searchQueries: 100,
    useRealModels: false, // Use mock data for faster testing
    cleanup: true,
  });

  // Run all benchmarks
  const results = await runner.runAll();

  // Print formatted report
  runner.printReport();

  // Export as JSON
  const json = runner.exportJSON();
  console.log(json);
}

main().catch(console.error);
```

Then run with Node.js (requires ES modules support):

```bash
# Option 1: Use .mjs extension
node my-benchmark.mjs

# Option 2: Add "type": "module" to package.json
node my-benchmark.js

# Option 3: Use tsx for TypeScript
npx tsx examples/benchmark-usage.ts
```

### Running Benchmarks in Browser

**Simple Demo (Recommended for Browser):**
```bash
npx serve examples
# Open http://localhost:8080/examples/benchmark-demo-simple.html
```

This standalone demo tests basic JavaScript performance without requiring the full VectorDB library.

**Full VectorDB Benchmarks:**

The VectorDB library is designed for browser environments. For comprehensive benchmarks:

1. **Integrate into your web application** with a bundler (webpack/vite/rollup)
2. **Use the test suite** which runs in a browser-like environment:
   ```bash
   npm test -- src/performance/Benchmark.test.ts --run
   ```
3. **Create a custom HTML page** based on `examples/benchmark-demo.html` and bundle it with your build tool

## Benchmark Configuration

### BenchmarkRunnerConfig

```typescript
interface BenchmarkRunnerConfig {
  // Dataset sizes to test (default: [100, 1000, 10000])
  datasetSizes?: number[];
  
  // Number of search queries per dataset (default: 100)
  searchQueries?: number;
  
  // Embedding model to use (default: 'Xenova/all-MiniLM-L6-v2')
  embeddingModel?: string;
  
  // Whether to test with real models or mock data (default: false)
  useRealModels?: boolean;
  
  // Whether to clean up after each test (default: true)
  cleanup?: boolean;
}
```

## Benchmark Types

### 1. Search Latency Benchmark

Measures search performance across different dataset sizes.

**Metrics:**
- Mean latency (ms)
- Median latency (ms)
- P95 latency (ms)
- P99 latency (ms)
- Min/Max latency (ms)

**Performance Targets:**
- 10K vectors: < 100ms mean latency
- 50K vectors: < 500ms mean latency
- 100K vectors: < 1000ms mean latency

**Example:**
```typescript
// Search Latency (10000 vectors)
// Mean: 45.23ms
// P95: 67.89ms
// P99: 89.12ms
```

### 2. Insertion Throughput Benchmark

Measures how many documents can be inserted per second.

**Metrics:**
- Operations per second
- Average latency per operation (ms)
- Total operations
- Total duration (ms)

**Performance Targets:**
- Single insert: > 100 ops/sec
- Batch insert (100 docs): > 10 batches/sec

**Example:**
```typescript
// Single Insert Throughput
// Ops/sec: 234.56
// Avg Latency: 4.26ms
```

### 3. Memory Usage Benchmark

Profiles memory consumption during operations.

**Metrics:**
- Peak memory (MB)
- Average memory (MB)
- Memory growth (MB)
- Min/Max memory (MB)
- Number of samples

**Performance Targets:**
- 50K vectors (768d): < 500MB peak memory
- Memory growth: < 200MB during 5K insertions

**Example:**
```typescript
// Memory Usage During Insertion
// Peak Memory: 234.56 MB
// Memory Growth: 123.45 MB
// Avg Memory: 198.76 MB
```

### 4. Batch Operations Benchmark

Measures performance of batch insertions with different batch sizes.

**Metrics:**
- Mean latency per batch
- Latency by batch size (10, 50, 100, 500)

**Example:**
```typescript
// Batch Insert (100 docs)
// Mean: 123.45ms
// Median: 118.23ms
```

### 5. Cache Performance Benchmark

Evaluates cache effectiveness and performance impact.

**Metrics:**
- Cold cache search time
- Warm cache search time
- Cache hit rate (%)
- Performance improvement (%)

**Example:**
```typescript
// Search (Cold Cache): 45.23ms
// Search (Warm Cache): 12.34ms
// Improvement: 72.7%
```

### 6. Model Load Time Benchmark

Measures embedding model initialization time.

**Metrics:**
- Mean load time (ms)
- Min/Max load time (ms)

**Performance Targets:**
- Cached model: < 3 seconds
- First load: < 10 seconds

## Interpreting Results

### Benchmark Report Format

```
================================================================================
VectorDB Performance Benchmark Report
================================================================================

Environment:
  Browser: Chrome 120
  Platform: MacIntel
  CPU Cores: 8
  Device Memory: 16 GB

--------------------------------------------------------------------------------
Search Latency (10000 vectors)
  Average search time for k=10 on 10000 vector dataset

  Metrics:
    iterations: 100
    min: 38.45
    max: 89.23
    mean: 45.67
    median: 44.12
    p95: 67.89
    p99: 78.34

--------------------------------------------------------------------------------
Single Insert Throughput
  Number of single document insertions per second

  Metrics:
    operations: 1234
    duration: 5000.12
    opsPerSecond: 246.78
    avgLatency: 4.05
    throughput: 246.78

================================================================================
```

### Performance Validation

The benchmark suite automatically validates against performance targets:

```typescript
// ✓ PASS: Search Latency (10K vectors)
//   Target: <100ms
//   Actual: 45.67ms

// ✓ PASS: Insert Throughput
//   Target: >100 ops/sec
//   Actual: 246.78 ops/sec

// ✓ PASS: Memory Usage
//   Target: <500MB
//   Actual: 234.56MB
```

## Custom Benchmarks

### Using the Benchmark Class Directly

```typescript
import { Benchmark } from '@vectordb/browser-vectordb';

const benchmark = new Benchmark();

// Simple timing benchmark
const result = await benchmark.run(
  'My Custom Benchmark',
  'Description of what is being tested',
  async () => {
    // Your code to benchmark
    await myFunction();
  },
  {
    iterations: 100,
    warmup: 5,
    collectMemory: true,
  }
);

console.log(`Mean: ${result.metrics.mean}ms`);
console.log(`P95: ${result.metrics.p95}ms`);
```

### Throughput Benchmark

```typescript
// Measure operations per second
const result = await benchmark.runThroughput(
  'Custom Throughput',
  'Measure ops/sec',
  async () => {
    await myOperation();
  },
  {
    duration: 5000, // Run for 5 seconds
    warmup: 10,     // 10 warmup iterations
  }
);

console.log(`Throughput: ${result.metrics.opsPerSecond} ops/sec`);
```

### Memory Profiling

```typescript
// Profile memory usage over time
const result = await benchmark.profileMemory(
  'Memory Profile',
  'Track memory during operation',
  async () => {
    // Your memory-intensive operation
    for (let i = 0; i < 1000; i++) {
      await processItem(i);
    }
  },
  {
    sampleInterval: 100, // Sample every 100ms
  }
);

console.log(`Peak Memory: ${result.metrics.peakMemory}MB`);
console.log(`Memory Growth: ${result.metrics.memoryGrowth}MB`);
```

## Browser Compatibility

### Memory Profiling

Memory profiling requires the `performance.memory` API, which is available in:
- Chrome/Edge (with `--enable-precise-memory-info` flag)
- Not available in Firefox or Safari

Without this API, memory metrics will not be collected, but other benchmarks will still work.

### WebGPU Acceleration

For benchmarks with real models:
- Chrome/Edge: WebGPU support available
- Firefox/Safari: Falls back to WASM

## Best Practices

### 1. Consistent Environment

Run benchmarks in a consistent environment:
- Close other tabs and applications
- Use the same browser and version
- Disable browser extensions
- Use incognito/private mode

### 2. Multiple Runs

Run benchmarks multiple times and compare results:
```typescript
const runs = [];
for (let i = 0; i < 3; i++) {
  const runner = new BenchmarkRunner(config);
  const results = await runner.runAll();
  runs.push(results);
}

// Compare results across runs
```

### 3. Warmup Iterations

Always use warmup iterations to:
- Allow JIT compilation
- Warm up caches
- Stabilize performance

### 4. Realistic Data

Use realistic test data:
- Varied document lengths
- Representative metadata
- Realistic query patterns

### 5. Monitor Memory

Watch for memory leaks:
- Memory should stabilize after operations
- Check for unbounded growth
- Verify cleanup after dispose

## Exporting Results

### JSON Export

```typescript
const runner = new BenchmarkRunner(config);
await runner.runAll();

// Export as JSON
const json = runner.exportJSON();

// Save to file or send to analytics service
await fetch('/api/benchmarks', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: json,
});
```

### CSV Export

```typescript
// Convert results to CSV format
function resultsToCSV(results) {
  const rows = [
    ['Name', 'Mean (ms)', 'P95 (ms)', 'P99 (ms)', 'Ops/Sec'],
  ];
  
  for (const result of results.results) {
    rows.push([
      result.name,
      result.metrics.mean || '',
      result.metrics.p95 || '',
      result.metrics.p99 || '',
      result.metrics.opsPerSecond || '',
    ]);
  }
  
  return rows.map(row => row.join(',')).join('\n');
}
```

## Continuous Benchmarking

### Automated Testing

Integrate benchmarks into CI/CD:

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm install
      - run: npm run benchmark
      - uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark-results.json
```

### Performance Regression Detection

Track performance over time:

```typescript
// Compare with baseline
const baseline = await loadBaseline();
const current = await runner.runAll();

for (const result of current.results) {
  const baselineResult = baseline.results.find(r => r.name === result.name);
  if (baselineResult) {
    const regression = (result.metrics.mean - baselineResult.metrics.mean) / baselineResult.metrics.mean;
    if (regression > 0.1) { // 10% regression threshold
      console.warn(`Performance regression detected in ${result.name}: ${(regression * 100).toFixed(2)}%`);
    }
  }
}
```

## Troubleshooting

### Benchmarks Running Slowly

- Disable real models: `useRealModels: false`
- Reduce dataset sizes: `datasetSizes: [100, 1000]`
- Reduce iterations: `searchQueries: 50`

### Memory Metrics Not Available

- Use Chrome/Edge with `--enable-precise-memory-info`
- Or accept that memory metrics won't be collected

### Inconsistent Results

- Increase warmup iterations
- Close other applications
- Run multiple times and average
- Check for background processes

### Out of Memory Errors

- Reduce dataset sizes
- Enable cleanup: `cleanup: true`
- Reduce cache sizes in VectorDB config

## Examples

See the following examples for complete implementations:

- `examples/benchmark-usage.ts` - Programmatic benchmark usage
- `examples/benchmark-demo.html` - Interactive browser demo
- `src/performance/Benchmark.test.ts` - Unit tests showing API usage

## API Reference

### Benchmark Class

```typescript
class Benchmark {
  // Run a timed benchmark
  async run<T>(
    name: string,
    description: string,
    fn: () => Promise<T>,
    options?: {
      warmup?: number;
      iterations?: number;
      collectMemory?: boolean;
    }
  ): Promise<BenchmarkResult>;

  // Run a throughput benchmark
  async runThroughput(
    name: string,
    description: string,
    fn: () => Promise<void>,
    options?: {
      duration?: number;
      warmup?: number;
    }
  ): Promise<BenchmarkResult>;

  // Profile memory usage
  async profileMemory(
    name: string,
    description: string,
    fn: () => Promise<void>,
    options?: {
      sampleInterval?: number;
    }
  ): Promise<BenchmarkResult>;

  // Get all results
  getResults(): BenchmarkResult[];

  // Get summary
  getSummary(): BenchmarkSuite;

  // Format as report
  formatReport(): string;

  // Export as JSON
  exportJSON(): string;

  // Clear results
  clear(): void;
}
```

### BenchmarkRunner Class

```typescript
class BenchmarkRunner {
  constructor(config?: BenchmarkRunnerConfig);

  // Run all benchmarks
  async runAll(): Promise<BenchmarkSuite>;

  // Get results
  getResults(): BenchmarkSuite;

  // Print formatted report
  printReport(): void;

  // Export as JSON
  exportJSON(): string;
}
```
