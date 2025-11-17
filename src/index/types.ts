/**
 * Index layer types
 */

import type { Filter } from '../storage/types';

export interface SearchQuery {
  vector?: Float32Array;
  text?: string;
  k: number;
  filter?: Filter;
  includeVectors?: boolean;
}

export interface SearchResult {
  id: string;
  score: number;
  metadata: Record<string, any>;
  vector?: Float32Array;
}

export interface IndexStats {
  vectorCount: number;
  dimensions: number;
  indexType: string;
  memoryUsage: number;
  lastUpdated: number;
}
