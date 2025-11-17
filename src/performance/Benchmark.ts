/**
 * Performance Benchmarking Suite
 * 
 * Comprehensive benchmarking for:
 * - Search latency across dataset sizes
 * - Insertion throughput
 * - Memory usage profiling
 * - Model load times
 */

export interface BenchmarkResult {
  name: string;
  description: string;
  metrics: {
    [key: string]: number | string;
  };
  timestamp: number;
  environment: BenchmarkEnvironment;
}

export interface BenchmarkEnvironment {
  browser: string;
  browserVersion: string;
  platform: string;
  hardwareConcurrency: number;
  deviceMemory?: number;
  connection?: string;
}

export interface BenchmarkSuite {
  name: string;
  results: BenchmarkResult[];
  summary: {
    totalTests: number;
    totalDuration: number;
    environment: BenchmarkEnvironment;
  };
}

/**
 * Utility class for running performance benchmarks
 */
export class Benchmark {
  private results: BenchmarkResult[] = [];
  private environment: BenchmarkEnvironment;

  constructor() {
    this.environment = this.detectEnvironment();
  }

  /**
   * Detect browser and system environment
   */
  private detectEnvironment(): BenchmarkEnvironment {
    const ua = navigator.userAgent;
    let browser = 'Unknown';
    let browserVersion = 'Unknown';

    // Detect browser
    if (ua.includes('Chrome') && !ua.includes('Edg')) {
      browser = 'Chrome';
      const match = ua.match(/Chrome\/(\d+)/);
      browserVersion = match ? match[1] : 'Unknown';
    } else if (ua.includes('Firefox')) {
      browser = 'Firefox';
      const match = ua.match(/Firefox\/(\d+)/);
      browserVersion = match ? match[1] : 'Unknown';
    } else if (ua.includes('Safari') && !ua.includes('Chrome')) {
      browser = 'Safari';
      const match = ua.match(/Version\/(\d+)/);
      browserVersion = match ? match[1] : 'Unknown';
    } else if (ua.includes('Edg')) {
      browser = 'Edge';
      const match = ua.match(/Edg\/(\d+)/);
      browserVersion = match ? match[1] : 'Unknown';
    }

    return {
      browser,
      browserVersion,
      platform: navigator.platform,
      hardwareConcurrency: navigator.hardwareConcurrency || 1,
      deviceMemory: (navigator as any).deviceMemory,
      connection: (navigator as any).connection?.effectiveType,
    };
  }

  /**
   * Run a benchmark function and measure performance
   */
  async run<T>(
    name: string,
    description: string,
    fn: () => Promise<T>,
    options: {
      warmup?: number;
      iterations?: number;
      collectMemory?: boolean;
    } = {}
  ): Promise<BenchmarkResult> {
    const { warmup = 0, iterations = 1, collectMemory = true } = options;

    // Warmup runs
    for (let i = 0; i < warmup; i++) {
      await fn();
    }

    // Force garbage collection if available
    if ((globalThis as any).gc) {
      (globalThis as any).gc();
    }

    // Collect initial memory
    const memoryBefore = collectMemory ? this.getMemoryUsage() : null;

    // Benchmark runs
    const durations: number[] = [];

    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await fn();
      const end = performance.now();
      durations.push(end - start);
    }

    // Collect final memory
    const memoryAfter = collectMemory ? this.getMemoryUsage() : null;

    // Calculate statistics
    const metrics: { [key: string]: number | string } = {
      iterations,
      min: Math.min(...durations),
      max: Math.max(...durations),
      mean: durations.reduce((a, b) => a + b, 0) / durations.length,
      median: this.calculateMedian(durations),
      p95: this.calculatePercentile(durations, 0.95),
      p99: this.calculatePercentile(durations, 0.99),
    };

    if (memoryBefore && memoryAfter) {
      metrics.memoryBefore = memoryBefore;
      metrics.memoryAfter = memoryAfter;
      metrics.memoryDelta = memoryAfter - memoryBefore;
    }

    const benchmarkResult: BenchmarkResult = {
      name,
      description,
      metrics,
      timestamp: Date.now(),
      environment: this.environment,
    };

    this.results.push(benchmarkResult);
    return benchmarkResult;
  }

  /**
   * Run a throughput benchmark (operations per second)
   */
  async runThroughput(
    name: string,
    description: string,
    fn: () => Promise<void>,
    options: {
      duration?: number; // milliseconds
      warmup?: number;
    } = {}
  ): Promise<BenchmarkResult> {
    const { duration = 5000, warmup = 0 } = options;

    // Warmup
    for (let i = 0; i < warmup; i++) {
      await fn();
    }

    // Force garbage collection
    if ((globalThis as any).gc) {
      (globalThis as any).gc();
    }

    // Run for specified duration
    const startTime = performance.now();
    let operations = 0;
    let totalDuration = 0;

    while (performance.now() - startTime < duration) {
      const opStart = performance.now();
      await fn();
      const opEnd = performance.now();
      operations++;
      totalDuration += opEnd - opStart;
    }

    const actualDuration = performance.now() - startTime;
    const opsPerSecond = (operations / actualDuration) * 1000;
    const avgLatency = totalDuration / operations;

    const benchmarkResult: BenchmarkResult = {
      name,
      description,
      metrics: {
        operations,
        duration: actualDuration,
        opsPerSecond,
        avgLatency,
        throughput: opsPerSecond,
      },
      timestamp: Date.now(),
      environment: this.environment,
    };

    this.results.push(benchmarkResult);
    return benchmarkResult;
  }

  /**
   * Measure memory usage over time during an operation
   */
  async profileMemory(
    name: string,
    description: string,
    fn: () => Promise<void>,
    options: {
      sampleInterval?: number; // milliseconds
    } = {}
  ): Promise<BenchmarkResult> {
    const { sampleInterval = 100 } = options;
    const samples: number[] = [];

    // Start sampling
    const samplingInterval = setInterval(() => {
      const memory = this.getMemoryUsage();
      if (memory !== null) {
        samples.push(memory);
      }
    }, sampleInterval);

    // Run the function
    const start = performance.now();
    await fn();
    const duration = performance.now() - start;

    // Stop sampling
    clearInterval(samplingInterval);

    // Calculate memory statistics
    const metrics: { [key: string]: number | string } = {
      duration,
      samples: samples.length,
      minMemory: Math.min(...samples),
      maxMemory: Math.max(...samples),
      avgMemory: samples.reduce((a, b) => a + b, 0) / samples.length,
      peakMemory: Math.max(...samples),
      memoryGrowth: samples[samples.length - 1] - samples[0],
    };

    const benchmarkResult: BenchmarkResult = {
      name,
      description,
      metrics,
      timestamp: Date.now(),
      environment: this.environment,
    };

    this.results.push(benchmarkResult);
    return benchmarkResult;
  }

  /**
   * Get current memory usage in MB
   */
  private getMemoryUsage(): number | null {
    if ((performance as any).memory) {
      return (performance as any).memory.usedJSHeapSize / 1024 / 1024;
    }
    return null;
  }

  /**
   * Calculate median of an array
   */
  private calculateMedian(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  /**
   * Calculate percentile of an array
   */
  private calculatePercentile(values: number[], percentile: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil(sorted.length * percentile) - 1;
    return sorted[Math.max(0, index)];
  }

  /**
   * Get all benchmark results
   */
  getResults(): BenchmarkResult[] {
    return this.results;
  }

  /**
   * Get a summary of all benchmarks
   */
  getSummary(): BenchmarkSuite {
    const totalDuration = this.results.reduce(
      (sum, r) => sum + (r.metrics.duration as number || 0),
      0
    );

    return {
      name: 'VectorDB Performance Benchmark',
      results: this.results,
      summary: {
        totalTests: this.results.length,
        totalDuration,
        environment: this.environment,
      },
    };
  }

  /**
   * Format results as a readable report
   */
  formatReport(): string {
    const lines: string[] = [];
    lines.push('='.repeat(80));
    lines.push('VectorDB Performance Benchmark Report');
    lines.push('='.repeat(80));
    lines.push('');
    lines.push(`Environment:`);
    lines.push(`  Browser: ${this.environment.browser} ${this.environment.browserVersion}`);
    lines.push(`  Platform: ${this.environment.platform}`);
    lines.push(`  CPU Cores: ${this.environment.hardwareConcurrency}`);
    if (this.environment.deviceMemory) {
      lines.push(`  Device Memory: ${this.environment.deviceMemory} GB`);
    }
    lines.push('');

    for (const result of this.results) {
      lines.push('-'.repeat(80));
      lines.push(`${result.name}`);
      lines.push(`  ${result.description}`);
      lines.push('');
      lines.push('  Metrics:');
      
      for (const [key, value] of Object.entries(result.metrics)) {
        const formattedValue = typeof value === 'number' 
          ? value.toFixed(2) 
          : value;
        lines.push(`    ${key}: ${formattedValue}`);
      }
      lines.push('');
    }

    lines.push('='.repeat(80));
    return lines.join('\n');
  }

  /**
   * Export results as JSON
   */
  exportJSON(): string {
    return JSON.stringify(this.getSummary(), null, 2);
  }

  /**
   * Clear all results
   */
  clear(): void {
    this.results = [];
  }
}
