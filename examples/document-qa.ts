/**
 * Document Q&A Example
 * 
 * This example demonstrates how to build a document question-answering system
 * using VectorDB and RAG pipeline for intelligent document retrieval and response generation.
 */

import { VectorDB, RAGPipelineManager, WllamaProvider } from '../src/index';

async function main() {
  console.log('=== Document Q&A System Example ===\n');

  // Step 1: Initialize VectorDB
  console.log('1. Initializing VectorDB...');
  const vectorDB = new VectorDB({
    storage: {
      dbName: 'document-qa-db',
      version: 1,
    },
    index: {
      dimensions: 384,
      metric: 'cosine',
      indexType: 'kdtree',
    },
    embedding: {
      model: 'Xenova/all-MiniLM-L6-v2',
      device: 'wasm',
      cache: true,
    },
  });

  await vectorDB.initialize();
  console.log('✓ VectorDB initialized\n');

  // Step 2: Load and chunk documents
  console.log('2. Loading and chunking documents...');
  
  // Simulate loading a large document and splitting it into chunks
  const documentChunks = [
    {
      text: 'Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.',
      metadata: {
        documentId: 'ai-intro',
        documentTitle: 'Introduction to Artificial Intelligence',
        chunkIndex: 0,
        section: 'Overview',
        pageNumber: 1,
        author: 'Dr. Jane Smith',
        publishDate: '2024-01-15',
      },
    },
    {
      text: 'Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.',
      metadata: {
        documentId: 'ai-intro',
        documentTitle: 'Introduction to Artificial Intelligence',
        chunkIndex: 1,
        section: 'Machine Learning',
        pageNumber: 2,
        author: 'Dr. Jane Smith',
        publishDate: '2024-01-15',
      },
    },
    {
      text: 'Deep Learning is a subset of machine learning that uses neural networks with multiple layers. These neural networks attempt to simulate the behavior of the human brain, allowing it to learn from large amounts of data.',
      metadata: {
        documentId: 'ai-intro',
        documentTitle: 'Introduction to Artificial Intelligence',
        chunkIndex: 2,
        section: 'Deep Learning',
        pageNumber: 3,
        author: 'Dr. Jane Smith',
        publishDate: '2024-01-15',
      },
    },
    {
      text: 'Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics.',
      metadata: {
        documentId: 'ai-intro',
        documentTitle: 'Introduction to Artificial Intelligence',
        chunkIndex: 3,
        section: 'Natural Language Processing',
        pageNumber: 4,
        author: 'Dr. Jane Smith',
        publishDate: '2024-01-15',
      },
    },
    {
      text: 'Computer Vision is a field of AI that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects.',
      metadata: {
        documentId: 'ai-intro',
        documentTitle: 'Introduction to Artificial Intelligence',
        chunkIndex: 4,
        section: 'Computer Vision',
        pageNumber: 5,
        author: 'Dr. Jane Smith',
        publishDate: '2024-01-15',
      },
    },
    {
      text: 'Reinforcement Learning is an area of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward. It is inspired by behaviorist psychology.',
      metadata: {
        documentId: 'ai-intro',
        documentTitle: 'Introduction to Artificial Intelligence',
        chunkIndex: 5,
        section: 'Reinforcement Learning',
        pageNumber: 6,
        author: 'Dr. Jane Smith',
        publishDate: '2024-01-15',
      },
    },
    {
      text: 'AI Ethics involves the moral implications of artificial intelligence systems. Key concerns include bias in AI algorithms, privacy issues, job displacement, and the potential for AI to be used in harmful ways.',
      metadata: {
        documentId: 'ai-intro',
        documentTitle: 'Introduction to Artificial Intelligence',
        chunkIndex: 6,
        section: 'AI Ethics',
        pageNumber: 7,
        author: 'Dr. Jane Smith',
        publishDate: '2024-01-15',
      },
    },
    {
      text: 'The future of AI includes developments in quantum computing, neuromorphic computing, and artificial general intelligence (AGI). These advances promise to revolutionize how we interact with technology.',
      metadata: {
        documentId: 'ai-intro',
        documentTitle: 'Introduction to Artificial Intelligence',
        chunkIndex: 7,
        section: 'Future of AI',
        pageNumber: 8,
        author: 'Dr. Jane Smith',
        publishDate: '2024-01-15',
      },
    },
  ];

  const chunkIds = await vectorDB.insertBatch(documentChunks);
  console.log(`✓ Inserted ${chunkIds.length} document chunks\n`);

  // Step 3: Initialize LLM Provider (optional for full RAG)
  console.log('3. Setting up RAG Pipeline...');
  console.log('Note: For full functionality, initialize an LLM provider.\n');

  const llmProvider = new WllamaProvider({
    modelUrl: 'YOUR_MODEL_URL_HERE', // Replace with actual model
    modelConfig: {
      n_ctx: 2048,
      n_threads: 4,
    },
  });

  // For this example, we'll demonstrate retrieval without actual LLM
  const ragPipeline = new RAGPipelineManager({
    vectorDB,
    llmProvider,
    embeddingGenerator: (vectorDB as any).embeddingGenerator,
    defaultContextTemplate: `[Document: {metadata.documentTitle}, Page {metadata.pageNumber}, Section: {metadata.section}]
{content}
(Relevance: {score})`,
    defaultMaxContextTokens: 1500,
  });

  console.log('✓ RAG Pipeline configured\n');

  // Step 4: Demonstrate Q&A with different question types
  console.log('4. Question Answering Examples...\n');

  const questions = [
    {
      question: 'What is Machine Learning?',
      type: 'Definition',
    },
    {
      question: 'How does Deep Learning differ from traditional Machine Learning?',
      type: 'Comparison',
    },
    {
      question: 'What are the main applications of Natural Language Processing?',
      type: 'Application',
    },
    {
      question: 'What ethical concerns are associated with AI?',
      type: 'Analysis',
    },
  ];

  for (const { question, type } of questions) {
    console.log(`Question Type: ${type}`);
    console.log(`Q: ${question}\n`);

    // Retrieve relevant chunks
    const results = await vectorDB.search({
      text: question,
      k: 3,
    });

    console.log('Retrieved Context:');
    results.forEach((result, index) => {
      console.log(`  ${index + 1}. [Page ${result.metadata.pageNumber}] ${result.metadata.section}`);
      console.log(`     Score: ${result.score.toFixed(4)}`);
      console.log(`     "${result.metadata.text.substring(0, 100)}..."`);
    });

    console.log('\nA: [In a real implementation, the LLM would generate an answer here]');
    console.log('   Based on the retrieved context, the answer would synthesize information');
    console.log('   from the relevant document sections.\n');
    console.log('─'.repeat(80));
    console.log();
  }

  // Step 5: Demonstrate citation and source tracking
  console.log('5. Citation and Source Tracking...\n');

  const citationQuery = 'Explain reinforcement learning';
  console.log(`Query: "${citationQuery}"\n`);

  const citationResults = await vectorDB.search({
    text: citationQuery,
    k: 2,
  });

  console.log('Answer with Citations:');
  console.log('Reinforcement learning is an area of machine learning where an agent');
  console.log('learns to make decisions by performing actions in an environment to');
  console.log('maximize cumulative reward [1].\n');

  console.log('Sources:');
  citationResults.forEach((result, index) => {
    console.log(`[${index + 1}] ${result.metadata.documentTitle}`);
    console.log(`    Author: ${result.metadata.author}`);
    console.log(`    Page: ${result.metadata.pageNumber}, Section: ${result.metadata.section}`);
    console.log(`    Published: ${result.metadata.publishDate}`);
    console.log(`    Relevance: ${(result.score * 100).toFixed(1)}%`);
    console.log();
  });

  // Step 6: Demonstrate filtering by document metadata
  console.log('6. Filtering by Document Metadata...\n');

  const filteredQuery = 'AI concepts';
  console.log(`Query: "${filteredQuery}" (filtered to specific sections)\n`);

  const filteredResults = await vectorDB.search({
    text: filteredQuery,
    k: 5,
    filter: {
      field: 'section',
      operator: 'in',
      value: ['Machine Learning', 'Deep Learning', 'Natural Language Processing'],
    },
  });

  console.log('Results from specific sections:');
  filteredResults.forEach((result, index) => {
    console.log(`  ${index + 1}. ${result.metadata.section} (Page ${result.metadata.pageNumber})`);
  });
  console.log();

  // Step 7: Demonstrate multi-document search
  console.log('7. Multi-Document Search...\n');

  // Add documents from another source
  const additionalDocs = [
    {
      text: 'Python is a high-level programming language widely used in AI and machine learning due to its simplicity and extensive libraries like TensorFlow and PyTorch.',
      metadata: {
        documentId: 'python-guide',
        documentTitle: 'Python for AI Development',
        chunkIndex: 0,
        section: 'Introduction',
        pageNumber: 1,
        author: 'John Doe',
        publishDate: '2024-02-01',
      },
    },
  ];

  await vectorDB.insertBatch(additionalDocs);

  const multiDocResults = await vectorDB.search({
    text: 'programming languages for AI',
    k: 3,
  });

  console.log('Results from multiple documents:');
  multiDocResults.forEach((result, index) => {
    console.log(`  ${index + 1}. ${result.metadata.documentTitle}`);
    console.log(`     Section: ${result.metadata.section}`);
    console.log(`     Score: ${result.score.toFixed(4)}`);
  });
  console.log();

  // Step 8: Demonstrate context window management
  console.log('8. Context Window Management...\n');

  const longQuery = 'Provide a comprehensive overview of AI including machine learning, deep learning, and their applications';
  console.log(`Query: "${longQuery}"\n`);

  const contextResults = await vectorDB.search({
    text: longQuery,
    k: 10, // Retrieve more chunks
  });

  console.log(`Retrieved ${contextResults.length} chunks`);
  console.log('Context window strategy:');
  console.log('  1. Rank by relevance score');
  console.log('  2. Deduplicate by document section');
  console.log('  3. Fit within token limit (1500 tokens)');
  console.log('  4. Maintain document order when possible');
  console.log();

  // Simulate token counting and selection
  let totalTokens = 0;
  const maxTokens = 1500;
  const selectedChunks: typeof contextResults = [];

  for (const result of contextResults) {
    const estimatedTokens = Math.ceil(result.metadata.text.length / 4); // Rough estimate
    if (totalTokens + estimatedTokens <= maxTokens) {
      selectedChunks.push(result);
      totalTokens += estimatedTokens;
    } else {
      break;
    }
  }

  console.log(`Selected ${selectedChunks.length} chunks (≈${totalTokens} tokens)\n`);

  // Step 9: Demonstrate answer confidence scoring
  console.log('9. Answer Confidence Scoring...\n');

  const confidenceQuery = 'What is quantum computing in AI?';
  console.log(`Query: "${confidenceQuery}"\n`);

  const confidenceResults = await vectorDB.search({
    text: confidenceQuery,
    k: 3,
  });

  const avgScore = confidenceResults.reduce((sum, r) => sum + r.score, 0) / confidenceResults.length;
  const confidence = avgScore > 0.7 ? 'High' : avgScore > 0.5 ? 'Medium' : 'Low';

  console.log(`Average relevance score: ${avgScore.toFixed(4)}`);
  console.log(`Confidence level: ${confidence}`);
  console.log();

  if (confidence === 'Low') {
    console.log('⚠️  Low confidence - the answer may not be well-supported by the documents');
  } else {
    console.log('✓ Sufficient confidence to provide an answer');
  }
  console.log();

  // Step 10: Performance metrics
  console.log('10. Performance Metrics...\n');

  const startTime = performance.now();
  await vectorDB.search({
    text: 'test query',
    k: 5,
  });
  const searchTime = performance.now() - startTime;

  console.log('System Performance:');
  console.log(`  Search latency: ${searchTime.toFixed(2)}ms`);
  console.log(`  Documents indexed: ${await vectorDB.size()}`);
  console.log(`  Embedding dimensions: 384`);
  console.log(`  Average chunk size: ~150 words`);
  console.log();

  // Step 11: Best practices
  console.log('11. Document Q&A Best Practices:\n');
  console.log('  ✓ Chunk documents into semantic units (paragraphs/sections)');
  console.log('  ✓ Include rich metadata (title, author, page, section)');
  console.log('  ✓ Use appropriate chunk size (100-500 words)');
  console.log('  ✓ Maintain document structure and context');
  console.log('  ✓ Implement citation tracking for transparency');
  console.log('  ✓ Filter by metadata to narrow search scope');
  console.log('  ✓ Monitor confidence scores for answer quality');
  console.log('  ✓ Handle multi-document scenarios gracefully');
  console.log();

  // Step 12: Use cases
  console.log('12. Document Q&A Use Cases:\n');
  console.log('  📚 Research Assistant: Answer questions from academic papers');
  console.log('  📖 Documentation Search: Find answers in technical docs');
  console.log('  📋 Legal Document Analysis: Extract information from contracts');
  console.log('  🏥 Medical Records: Query patient history and reports');
  console.log('  📰 News Analysis: Summarize and answer questions about articles');
  console.log('  📊 Business Intelligence: Query reports and analytics');
  console.log();

  // Cleanup
  console.log('13. Cleanup...');
  await llmProvider.dispose();
  await vectorDB.dispose();
  console.log('✓ Resources cleaned up\n');

  console.log('=== Document Q&A Example Complete ===');
  console.log('\nNext Steps:');
  console.log('1. Implement document chunking strategy');
  console.log('2. Add support for various document formats (PDF, DOCX, etc.)');
  console.log('3. Build a web interface for document upload and Q&A');
  console.log('4. Integrate with a real LLM for answer generation');
  console.log('5. Add support for follow-up questions and conversation context');
}

// Helper function to chunk documents
function chunkDocument(text: string, chunkSize: number = 500, overlap: number = 50): string[] {
  const chunks: string[] = [];
  const words = text.split(/\s+/);
  
  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    const chunk = words.slice(i, i + chunkSize).join(' ');
    chunks.push(chunk);
  }
  
  return chunks;
}

// Helper function to estimate tokens
function estimateTokens(text: string): number {
  // Rough estimate: 1 token ≈ 4 characters
  return Math.ceil(text.length / 4);
}

// Run the example
main().catch(console.error);
