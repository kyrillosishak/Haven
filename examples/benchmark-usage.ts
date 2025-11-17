/**
 * Example: Running Performance Benchmarks
 * 
 * This example demonstrates how to run comprehensive performance benchmarks
 * on the VectorDB system, including:
 * - Search latency across different dataset sizes
 * - Insertion throughput
 * - Memory usage profiling
 * - Cache performance
 * 
 * HOW TO RUN:
 * 
 * Since VectorDB is a browser library, you have these options:
 * 
 * 1. Simple Browser Demo (Easiest):
 *    npx serve examples
 *    Open: http://localhost:3000/benchmark-demo-simple.html
 * 
 * 2. Run Tests (Validates Implementation):
 *    npm test -- src/performance/Benchmark.test.ts --run
 * 
 * 3. Integrate into Your Web App:
 *    Import this code into your bundled application (webpack/vite/rollup)
 * 
 * Note: This file is provided as a reference implementation.
 */

import { BenchmarkRunner } from '../src/performance/BenchmarkRunner';

async function main() {
  console.log('VectorDB Performance Benchmark Suite');
  console.log('=====================================\n');

  // Create benchmark runner with configuration
  const runner = new BenchmarkRunner({
    // Test with different dataset sizes
    datasetSizes: [100, 1000, 10000],
    
    // Number of search queries to run per dataset
    searchQueries: 100,
    
    // Embedding model to use (set to false to use mock data for faster testing)
    useRealModels: false,
    
    // Clean up test databases after benchmarks
    cleanup: true,
  });

  // Run all benchmarks
  console.log('Starting benchmark suite...\n');
  const results = await runner.runAll();

  // Print formatted report
  console.log('\n' + '='.repeat(80));
  console.log('BENCHMARK RESULTS');
  console.log('='.repeat(80) + '\n');
  
  runner.printReport();

  // Export results as JSON
  const jsonResults = runner.exportJSON();
  console.log('\nJSON Results:');
  console.log(jsonResults);

  // Analyze specific metrics
  console.log('\n' + '='.repeat(80));
  console.log('KEY METRICS SUMMARY');
  console.log('='.repeat(80) + '\n');

  for (const result of results.results) {
    if (result.name.includes('Search Latency')) {
      console.log(`${result.name}:`);
      console.log(`  Mean: ${(result.metrics.mean as number).toFixed(2)}ms`);
      console.log(`  P95: ${(result.metrics.p95 as number).toFixed(2)}ms`);
      console.log(`  P99: ${(result.metrics.p99 as number).toFixed(2)}ms`);
      console.log('');
    } else if (result.name.includes('Throughput')) {
      console.log(`${result.name}:`);
      console.log(`  Ops/sec: ${(result.metrics.opsPerSecond as number).toFixed(2)}`);
      console.log(`  Avg Latency: ${(result.metrics.avgLatency as number).toFixed(2)}ms`);
      console.log('');
    } else if (result.name.includes('Memory')) {
      console.log(`${result.name}:`);
      if (result.metrics.peakMemory) {
        console.log(`  Peak Memory: ${(result.metrics.peakMemory as number).toFixed(2)}MB`);
        console.log(`  Memory Growth: ${(result.metrics.memoryGrowth as number).toFixed(2)}MB`);
      }
      console.log('');
    }
  }

  // Performance targets validation
  console.log('='.repeat(80));
  console.log('PERFORMANCE TARGETS VALIDATION');
  console.log('='.repeat(80) + '\n');

  const searchLatency10K = results.results.find(
    r => r.name === 'Search Latency (10000 vectors)'
  );

  if (searchLatency10K) {
    const target = 100; // 100ms target for 10K vectors
    const actual = searchLatency10K.metrics.mean as number;
    const status = actual < target ? '✓ PASS' : '✗ FAIL';
    console.log(`Search Latency (10K vectors): ${status}`);
    console.log(`  Target: <${target}ms`);
    console.log(`  Actual: ${actual.toFixed(2)}ms`);
    console.log('');
  }

  const insertThroughput = results.results.find(
    r => r.name === 'Single Insert Throughput'
  );

  if (insertThroughput) {
    const target = 100; // 100 ops/sec minimum
    const actual = insertThroughput.metrics.opsPerSecond as number;
    const status = actual > target ? '✓ PASS' : '✗ FAIL';
    console.log(`Insert Throughput: ${status}`);
    console.log(`  Target: >${target} ops/sec`);
    console.log(`  Actual: ${actual.toFixed(2)} ops/sec`);
    console.log('');
  }

  const memoryUsage = results.results.find(
    r => r.name === 'Memory Usage During Insertion'
  );

  if (memoryUsage) {
    const target = 500; // 500MB target
    const actual = memoryUsage.metrics.peakMemory as number;
    const status = actual < target ? '✓ PASS' : '✗ FAIL';
    console.log(`Memory Usage: ${status}`);
    console.log(`  Target: <${target}MB`);
    console.log(`  Actual: ${actual.toFixed(2)}MB`);
    console.log('');
  }

  console.log('='.repeat(80));
  console.log('\nBenchmark suite complete!');
}

// Run the example
main().catch(console.error);
