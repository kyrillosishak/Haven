# Metadata Filtering

The VectorDB supports powerful metadata filtering capabilities for both storage queries and vector search operations. You can filter records using simple conditions or complex compound filters with AND/OR logic.

## Filter Types

### Simple Metadata Filter

A simple filter applies a single condition to a metadata field:

```typescript
interface MetadataFilter {
  field: string;
  operator: 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte' | 'in' | 'contains';
  value: any;
}
```

**Supported Operators:**
- `eq`: Equal to
- `ne`: Not equal to
- `gt`: Greater than
- `gte`: Greater than or equal to
- `lt`: Less than
- `lte`: Less than or equal to
- `in`: Value is in array
- `contains`: Array or string contains value

### Compound Filter

A compound filter combines multiple filters using AND/OR logic:

```typescript
interface CompoundFilter {
  operator: 'and' | 'or';
  filters: (MetadataFilter | CompoundFilter)[];
}
```

Compound filters can be nested to create complex filtering logic.

## Usage Examples

### Simple Filters

```typescript
// Equality filter
const filter1: MetadataFilter = {
  field: 'category',
  operator: 'eq',
  value: 'technology'
};

// Range filter
const filter2: MetadataFilter = {
  field: 'score',
  operator: 'gte',
  value: 80
};

// Contains filter (for arrays)
const filter3: MetadataFilter = {
  field: 'tags',
  operator: 'contains',
  value: 'important'
};

// In filter
const filter4: MetadataFilter = {
  field: 'status',
  operator: 'in',
  value: ['active', 'pending']
};
```

### Compound AND Filter

Match records that satisfy ALL conditions:

```typescript
const andFilter: CompoundFilter = {
  operator: 'and',
  filters: [
    { field: 'category', operator: 'eq', value: 'technology' },
    { field: 'score', operator: 'gte', value: 80 },
    { field: 'tags', operator: 'contains', value: 'featured' }
  ]
};

// This matches records where:
// - category is 'technology' AND
// - score is >= 80 AND
// - tags contains 'featured'
```

### Compound OR Filter

Match records that satisfy ANY condition:

```typescript
const orFilter: CompoundFilter = {
  operator: 'or',
  filters: [
    { field: 'category', operator: 'eq', value: 'technology' },
    { field: 'category', operator: 'eq', value: 'science' },
    { field: 'priority', operator: 'eq', value: 'high' }
  ]
};

// This matches records where:
// - category is 'technology' OR
// - category is 'science' OR
// - priority is 'high'
```

### Nested Compound Filters

Create complex logic by nesting compound filters:

```typescript
const nestedFilter: CompoundFilter = {
  operator: 'and',
  filters: [
    // First condition: category must be tech OR science
    {
      operator: 'or',
      filters: [
        { field: 'category', operator: 'eq', value: 'technology' },
        { field: 'category', operator: 'eq', value: 'science' }
      ]
    },
    // Second condition: score must be >= 80
    { field: 'score', operator: 'gte', value: 80 },
    // Third condition: must have 'featured' tag
    { field: 'tags', operator: 'contains', value: 'featured' }
  ]
};

// This matches records where:
// - (category is 'technology' OR category is 'science') AND
// - score is >= 80 AND
// - tags contains 'featured'
```

## Using Filters with Storage

Filter records directly from storage:

```typescript
import { IndexedDBStorage } from '@vectordb/browser-vectordb';

const storage = new IndexedDBStorage({ dbName: 'mydb' });
await storage.initialize();

// Simple filter
const results = await storage.filter({
  field: 'category',
  operator: 'eq',
  value: 'technology'
});

// Compound filter
const complexResults = await storage.filter({
  operator: 'and',
  filters: [
    { field: 'category', operator: 'eq', value: 'technology' },
    { field: 'score', operator: 'gte', value: 80 }
  ]
});
```

## Using Filters with Vector Search

Apply filters during similarity search to narrow results:

```typescript
import { VectorDB } from '@vectordb/browser-vectordb';

const db = new VectorDB({
  storage: { dbName: 'mydb' },
  index: { dimensions: 384, metric: 'cosine' },
  embedding: { model: 'Xenova/all-MiniLM-L6-v2', device: 'wasm' }
});

await db.initialize();

// Search with simple filter
const results = await db.search({
  text: 'machine learning',
  k: 10,
  filter: {
    field: 'category',
    operator: 'eq',
    value: 'technology'
  }
});

// Search with compound filter
const filteredResults = await db.search({
  text: 'artificial intelligence',
  k: 10,
  filter: {
    operator: 'and',
    filters: [
      { field: 'category', operator: 'eq', value: 'technology' },
      { field: 'score', operator: 'gte', value: 80 },
      { field: 'tags', operator: 'contains', value: 'ai' }
    ]
  }
});
```

## Performance Considerations

### Post-Search Filtering

The current implementation applies filters **after** vector search. This means:

1. The index performs k-NN search to find similar vectors
2. Results are retrieved from storage
3. Filters are applied to narrow down results
4. Final filtered results are returned

To account for filtering, the search retrieves more candidates than requested (3x multiplier when filters are present).

### IndexedDB Indexes

The storage layer creates IndexedDB indexes for common metadata fields:
- `timestamp`: For LRU eviction
- `metadata.tags`: For efficient tag-based filtering (multiEntry index)

You can optimize filtering performance by:
1. Using indexed fields when possible
2. Placing more selective filters first in compound AND filters
3. Limiting the number of nested compound filters

### Best Practices

1. **Use specific filters**: More specific filters reduce the number of records to process
2. **Combine filters efficiently**: In AND filters, place the most selective condition first
3. **Avoid deep nesting**: Keep compound filter nesting to 2-3 levels maximum
4. **Index frequently filtered fields**: Consider adding custom IndexedDB indexes for fields you filter often

## Type Safety

All filter types are fully typed in TypeScript:

```typescript
import type { 
  MetadataFilter, 
  CompoundFilter, 
  Filter 
} from '@vectordb/browser-vectordb';

// Filter is a union type of MetadataFilter | CompoundFilter
const myFilter: Filter = {
  operator: 'and',
  filters: [
    { field: 'category', operator: 'eq', value: 'tech' }
  ]
};
```

## Requirements Satisfied

This implementation satisfies the following requirements from the specification:

- **3.1**: Metadata filtering with similarity search
- **3.2**: Equality filters on string, number, and boolean fields
- **3.3**: Range filters (gt, gte, lt, lte) on numeric and date fields
- **3.4**: Array membership filters (in, contains)
- **3.5**: Compound filters with AND/OR logic
