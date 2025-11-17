/**
 * Tests for Benchmark utility
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { Benchmark } from './Benchmark';

describe('Benchmark', () => {
  let benchmark: Benchmark;

  beforeEach(() => {
    benchmark = new Benchmark();
  });

  describe('Environment Detection', () => {
    it('should detect browser environment', () => {
      const results = benchmark.getResults();
      const summary = benchmark.getSummary();
      
      expect(summary.summary.environment).toBeDefined();
      expect(summary.summary.environment.browser).toBeDefined();
      expect(summary.summary.environment.platform).toBeDefined();
      expect(summary.summary.environment.hardwareConcurrency).toBeGreaterThan(0);
    });
  });

  describe('run()', () => {
    it('should measure execution time', async () => {
      const result = await benchmark.run(
        'Test Benchmark',
        'Test description',
        async () => {
          await new Promise(resolve => setTimeout(resolve, 10));
        },
        { iterations: 5 }
      );

      expect(result.name).toBe('Test Benchmark');
      expect(result.description).toBe('Test description');
      expect(result.metrics.iterations).toBe(5);
      expect(result.metrics.mean).toBeGreaterThan(0);
      expect(result.metrics.min).toBeGreaterThan(0);
      expect(result.metrics.max).toBeGreaterThan(0);
      expect(result.metrics.median).toBeGreaterThan(0);
    });

    it('should calculate percentiles correctly', async () => {
      const result = await benchmark.run(
        'Percentile Test',
        'Test percentile calculation',
        async () => {
          await new Promise(resolve => setTimeout(resolve, Math.random() * 20));
        },
        { iterations: 100 }
      );

      expect(result.metrics.p95).toBeDefined();
      expect(result.metrics.p99).toBeDefined();
      expect(result.metrics.p99 as number).toBeGreaterThanOrEqual(result.metrics.p95 as number);
      expect(result.metrics.p95 as number).toBeGreaterThanOrEqual(result.metrics.median as number);
    });

    it('should support warmup iterations', async () => {
      let counter = 0;
      
      const result = await benchmark.run(
        'Warmup Test',
        'Test with warmup',
        async () => {
          counter++;
        },
        { warmup: 3, iterations: 5 }
      );

      expect(counter).toBe(8); // 3 warmup + 5 iterations
      expect(result.metrics.iterations).toBe(5);
    });

    it('should collect memory metrics when available', async () => {
      const result = await benchmark.run(
        'Memory Test',
        'Test memory collection',
        async () => {
          // Create array to use memory
          const arr: number[] = new Array(1000).fill(0);
          // Use the array to prevent optimization
          return arr.length;
        },
        { iterations: 1, collectMemory: true }
      );

      // Memory metrics may not be available in all environments
      if ((performance as any).memory) {
        expect(result.metrics.memoryBefore).toBeDefined();
        expect(result.metrics.memoryAfter).toBeDefined();
        expect(result.metrics.memoryDelta).toBeDefined();
      }
    });
  });

  describe('runThroughput()', () => {
    it('should measure operations per second', async () => {
      let counter = 0;
      
      const result = await benchmark.runThroughput(
        'Throughput Test',
        'Test throughput measurement',
        async () => {
          counter++;
        },
        { duration: 1000, warmup: 0 }
      );

      expect(result.metrics.operations).toBeGreaterThan(0);
      expect(result.metrics.opsPerSecond).toBeGreaterThan(0);
      expect(result.metrics.avgLatency).toBeGreaterThan(0);
      expect(result.metrics.throughput).toBe(result.metrics.opsPerSecond);
    });

    it('should respect duration parameter', async () => {
      const duration = 500;
      const start = performance.now();
      
      await benchmark.runThroughput(
        'Duration Test',
        'Test duration',
        async () => {
          await new Promise(resolve => setTimeout(resolve, 10));
        },
        { duration, warmup: 0 }
      );

      const elapsed = performance.now() - start;
      expect(elapsed).toBeGreaterThanOrEqual(duration);
      expect(elapsed).toBeLessThan(duration + 200); // Allow some overhead
    });
  });

  describe('profileMemory()', () => {
    it('should profile memory usage over time', async () => {
      const result = await benchmark.profileMemory(
        'Memory Profile Test',
        'Test memory profiling',
        async () => {
          const arrays: number[][] = [];
          for (let i = 0; i < 10; i++) {
            arrays.push(new Array(1000).fill(i));
            await new Promise(resolve => setTimeout(resolve, 50));
          }
        },
        { sampleInterval: 50 }
      );

      expect(result.metrics.duration).toBeGreaterThan(0);
      
      // Memory metrics may not be available in all environments
      if ((performance as any).memory) {
        expect(result.metrics.samples).toBeGreaterThan(0);
        expect(result.metrics.minMemory).toBeDefined();
        expect(result.metrics.maxMemory).toBeDefined();
        expect(result.metrics.avgMemory).toBeDefined();
        expect(result.metrics.peakMemory).toBeDefined();
      } else {
        // If memory API not available, samples will be 0
        expect(result.metrics.samples).toBe(0);
      }
    });
  });

  describe('Result Management', () => {
    it('should store benchmark results', async () => {
      await benchmark.run('Test 1', 'First test', async () => {});
      await benchmark.run('Test 2', 'Second test', async () => {});

      const results = benchmark.getResults();
      expect(results).toHaveLength(2);
      expect(results[0].name).toBe('Test 1');
      expect(results[1].name).toBe('Test 2');
    });

    it('should generate summary', async () => {
      await benchmark.run('Test 1', 'First test', async () => {});
      await benchmark.run('Test 2', 'Second test', async () => {});

      const summary = benchmark.getSummary();
      expect(summary.name).toBe('VectorDB Performance Benchmark');
      expect(summary.results).toHaveLength(2);
      expect(summary.summary.totalTests).toBe(2);
      expect(summary.summary.environment).toBeDefined();
    });

    it('should format report', async () => {
      await benchmark.run('Test Benchmark', 'Test description', async () => {
        await new Promise(resolve => setTimeout(resolve, 10));
      });

      const report = benchmark.formatReport();
      expect(report).toContain('VectorDB Performance Benchmark Report');
      expect(report).toContain('Test Benchmark');
      expect(report).toContain('Test description');
      expect(report).toContain('Metrics:');
    });

    it('should export JSON', async () => {
      await benchmark.run('Test', 'Description', async () => {});

      const json = benchmark.exportJSON();
      const parsed = JSON.parse(json);
      
      expect(parsed.name).toBe('VectorDB Performance Benchmark');
      expect(parsed.results).toHaveLength(1);
      expect(parsed.summary).toBeDefined();
    });

    it('should clear results', async () => {
      await benchmark.run('Test', 'Description', async () => {});
      expect(benchmark.getResults()).toHaveLength(1);

      benchmark.clear();
      expect(benchmark.getResults()).toHaveLength(0);
    });
  });

  describe('Statistics Calculation', () => {
    it('should calculate median correctly for odd number of values', async () => {
      const result = await benchmark.run(
        'Median Odd Test',
        'Test median with odd count',
        async () => {
          // Use consistent timing
          await new Promise(resolve => setTimeout(resolve, 5));
        },
        { iterations: 5 }
      );

      // Median should be between min and max
      expect(result.metrics.median as number).toBeGreaterThanOrEqual(result.metrics.min as number);
      expect(result.metrics.median as number).toBeLessThanOrEqual(result.metrics.max as number);
    });

    it('should calculate median correctly for even number of values', async () => {
      const result = await benchmark.run(
        'Median Even Test',
        'Test median with even count',
        async () => {
          // Use consistent timing
          await new Promise(resolve => setTimeout(resolve, 5));
        },
        { iterations: 4 }
      );

      // Median should be between min and max
      expect(result.metrics.median as number).toBeGreaterThanOrEqual(result.metrics.min as number);
      expect(result.metrics.median as number).toBeLessThanOrEqual(result.metrics.max as number);
    });
  });
});
