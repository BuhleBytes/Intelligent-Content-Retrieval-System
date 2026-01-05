"""
Part 4: Vector Database Implementation
Intelligent Content Retrieval System

This script stores embeddings in ChromaDB and implements semantic search
with comprehensive testing and validation.

Configuration:
- Vector Database: ChromaDB
- Distance Metric: Cosine similarity
- Embedding Model: all-mpnet-base-v2 (768 dimensions)
- Storage Path: data/chromadb/
- Batch Size: 100 chunks

Author: Buhle Mlandu
"""

import chromadb
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple


def load_data() -> Tuple[List[Dict], np.ndarray]:
    """
    Load chunks and embeddings from previous parts.
    
    Returns:
        tuple: (chunks_data, embeddings)
        
    Raises:
        FileNotFoundError: If required data files don't exist
        ValueError: If data is corrupted or invalid
    """
    try:
        # Load chunks from Part 2
        chunks_file = "data/processed/all_chunks.json"
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_file}\n"
                "Please run Part 2 first!"
            )
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        if not chunks_data:
            raise ValueError("Chunks file is empty!")
        
        # Load embeddings from Part 3
        embeddings_file = "data/embeddings/embeddings.npz"
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_file}\n"
                "Please run Part 3 first!"
            )
        
        embeddings_data = np.load(embeddings_file)
        embeddings = embeddings_data['embeddings']
        
        # Verify data consistency
        if len(chunks_data) != len(embeddings):
            raise ValueError(
                f"Data mismatch: {len(chunks_data)} chunks but "
                f"{len(embeddings)} embeddings!"
            )
        
        return chunks_data, embeddings
    
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        raise
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        raise


def create_chromadb(chunks_data: List[Dict], embeddings: np.ndarray) -> chromadb.Collection:
    """
    Create and populate ChromaDB collection with batching.
    
    Args:
        chunks_data: List of chunk dictionaries from Part 2
        embeddings: NumPy array of embeddings from Part 3
        
    Returns:
        chromadb.Collection: Populated collection
        
    Notes:
        - Uses batching to handle ChromaDB limits
        - Configures cosine similarity for distance metric
        - Data persists to disk automatically
    """
    # Initialize client with persistent storage
    client = chromadb.PersistentClient(path="data/chromadb")
    
    # Create or get collection with cosine similarity
    collection = client.get_or_create_collection(
        name="intelligent_content_retrieval",
        metadata={
            "hnsw:space": "cosine",  # Distance metric
            "description": "Multi-domain content for semantic search",
            "model": "all-mpnet-base-v2",
            "dimensions": "768",
            "created_date": datetime.now().isoformat()
        }
    )
    
    # Check if collection already has data
    existing_count = collection.count()
    if existing_count > 0:
        print(f"    ⚠️  Collection already contains {existing_count} documents")
        user_input = input("    Delete and rebuild? (yes/no): ").strip().lower()
        
        if user_input == 'yes':
            client.delete_collection("intelligent_content_retrieval")
            collection = client.create_collection(
                name="intelligent_content_retrieval",
                metadata={
                    "hnsw:space": "cosine",
                    "description": "Multi-domain content for semantic search",
                    "model": "all-mpnet-base-v2",
                    "dimensions": "768",
                    "created_date": datetime.now().isoformat()
                }
            )
            print("    ✓ Collection deleted and recreated")
        else:
            print("    ✓ Using existing collection")
            return collection
    
    # Add data in batches (ChromaDB has limits on single operations)
    batch_size = 100
    total_chunks = len(chunks_data)
    
    print(f"    Adding {total_chunks} chunks in batches of {batch_size}...")
    
    for i in range(0, total_chunks, batch_size):
        batch_end = min(i + batch_size, total_chunks)
        
        # Prepare batch data
        batch_ids = [chunk['chunk_id'] for chunk in chunks_data[i:batch_end]]
        batch_embeddings = embeddings[i:batch_end].tolist()
        batch_documents = [chunk['text'] for chunk in chunks_data[i:batch_end]]
        batch_metadatas = [chunk['metadata'] for chunk in chunks_data[i:batch_end]]
        
        # Add to collection
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
        
        print(f"    ✓ Batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}: "
              f"Added chunks {i} to {batch_end-1}")
    
    print(f"    ✓ Successfully added {total_chunks} chunks to ChromaDB!")
    
    return collection


def verify_persistence() -> chromadb.Collection:
    """
    Verify that data persists across sessions.
    
    Returns:
        chromadb.Collection: Existing collection loaded from disk
        
    Raises:
        ValueError: If collection doesn't exist
    """
    try:
        client = chromadb.PersistentClient(path="data/chromadb")
        collection = client.get_collection("intelligent_content_retrieval")
        
        count = collection.count()
        print(f"    ✓ Collection loaded from disk")
        print(f"    ✓ Contains {count} documents")
        print(f"    ✓ Storage location: data/chromadb/")
        
        return collection
    
    except Exception as e:
        raise ValueError(
            f"Collection not found or corrupted: {e}\n"
            "You may need to recreate the database."
        )


def display_database_stats(collection: chromadb.Collection) -> None:
    """
    Display comprehensive database statistics.
    
    Args:
        collection: ChromaDB collection to analyze
    """
    print("\n" + "="*70)
    print("DATABASE STATISTICS")
    print("="*70)
    
    # Get basic stats
    count = collection.count()
    metadata = collection.metadata
    
    # Get sample data
    sample = collection.peek(limit=5)
    
    print(f"\nCollection Information:")
    print(f"  Name: {collection.name}")
    print(f"  Total documents: {count}")
    print(f"  Distance metric: {metadata.get('hnsw:space', 'unknown')}")
    print(f"  Embedding model: {metadata.get('model', 'unknown')}")
    print(f"  Embedding dimensions: {metadata.get('dimensions', 'unknown')}")
    
    if 'created_date' in metadata:
        print(f"  Created: {metadata['created_date']}")
    
    # Analyze categories
    print(f"\nSample Documents (first 5):")
    for i, (doc_id, doc_text, doc_meta) in enumerate(zip(
        sample['ids'],
        sample['documents'],
        sample['metadatas']
    ), 1):
        print(f"  {i}. ID: {doc_id}")
        print(f"     Category: {doc_meta.get('source_category', 'N/A')}")
        print(f"     Preview: {doc_text[:80]}...")
    
    # Category distribution (requires querying all)
    print(f"\nStorage Information:")
    print(f"  Database path: data/chromadb/")
    print(f"  Persistence: ✓ Enabled (PersistentClient)")
    
    print("="*70 + "\n")


def semantic_search(
    collection: chromadb.Collection, 
    model: SentenceTransformer, 
    query_text: str, 
    n_results: int = 5,
    filter_category: str = None
) -> Dict:
    """
    Perform semantic search with the correct embedding model.
    
    Args:
        collection: ChromaDB collection to search
        model: SentenceTransformer model (must match Part 3)
        query_text: Natural language query
        n_results: Number of results to return (default: 5)
        filter_category: Optional category filter (e.g., "News", "Educational")
        
    Returns:
        dict: Search results with documents, metadata, and distances
        
    Notes:
        - Generates query embedding using same model as Part 3
        - Embeddings are normalized for cosine similarity
        - Lower distance = higher similarity
    """
    # Generate query embedding with YOUR model (not ChromaDB's default)
    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True  # Must match Part 3
    )
    
    # Prepare query parameters
    query_params = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": n_results
    }
    
    # Add category filter if specified
    if filter_category:
        query_params["where"] = {"source_category": filter_category}
    
    # Query ChromaDB
    results = collection.query(**query_params)
    
    return results


def display_results(results: Dict, query_text: str) -> None:
    """
    Display search results in a formatted, readable way.
    
    Args:
        results: Results dictionary from semantic_search()
        query_text: Original query for context
    """
    print(f"\n{'='*70}")
    print(f"QUERY: {query_text}")
    print(f"{'='*70}\n")
    
    # Check if results are empty
    if not results['documents'][0]:
        print("No results found!")
        return
    
    # Display each result
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        # Convert distance to similarity score (0-1, higher is better)
        similarity = 1 - distance
        
        print(f"--- Result {i} (Similarity: {similarity:.3f}) ---")
        print(f"Category: {metadata['source_category']}")
        print(f"Source: {metadata['source_url'][:60]}...")
        print(f"Domain: {metadata.get('source_domain', 'N/A')}")
        print(f"Chunk: {metadata.get('chunk_index', 'N/A')} / "
              f"{metadata.get('total_chunks_from_source', 'N/A')}")
        print(f"\nText Preview:")
        print(f"{doc[:300]}...")
        print()


def run_test_queries(collection: chromadb.Collection, model: SentenceTransformer) -> None:
    """
    Run predefined test queries for demonstration and validation.
    
    Args:
        collection: ChromaDB collection to search
        model: SentenceTransformer model for embedding generation
        
    Notes:
        Includes 5+ diverse queries as required by assignment:
        - Technical questions
        - Conceptual queries
        - How-to questions
        - Domain-specific questions
    """
    print("\n" + "="*70)
    print("RUNNING TEST QUERIES")
    print("="*70)
    print("\nExecuting 5 predefined diverse queries...\n")
    
    # Define test queries (diverse as per assignment requirements)
    test_queries = [
        {
            "query": "What is machine learning?",
            "category": "Technical/Educational",
            "type": "Definition query"
        },
        {
            "query": "How do neural networks learn from data?",
            "category": "Technical",
            "type": "Conceptual/How-to query"
        },
        {
            "query": "Explain the difference between supervised and unsupervised learning",
            "category": "Educational",
            "type": "Comparison query"
        },
        {
            "query": "What are the applications of artificial intelligence in healthcare?",
            "category": "Domain-specific",
            "type": "Application query"
        },
        {
            "query": "How does gradient descent optimize neural networks?",
            "category": "Technical",
            "type": "Procedural query"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'─'*70}")
        print(f"TEST QUERY {i}/{len(test_queries)}")
        print(f"Type: {test['type']}")
        print(f"Expected Category: {test['category']}")
        print(f"{'─'*70}")
        
        # Perform search
        results = semantic_search(
            collection, 
            model, 
            test['query'], 
            n_results=3  # Show top 3 for each test query
        )
        
        # Display results
        display_results(results, test['query'])
        
        # Pause between queries for readability
        if i < len(test_queries):
            input("Press Enter to continue to next test query...\n")
    
    print("\n" + "="*70)
    print("✓ All test queries completed!")
    print("="*70 + "\n")


def interactive_search(collection: chromadb.Collection, model: SentenceTransformer) -> None:
    """
    Interactive search interface for user queries.
    
    Args:
        collection: ChromaDB collection to search
        model: SentenceTransformer model for embedding generation
    """
    print("\n" + "="*70)
    print("INTERACTIVE SEARCH MODE")
    print("="*70)
    print("\nCommands:")
    print("  - Enter any question to search")
    print("  - Type 'filter:News' to search only News category")
    print("  - Type 'quit', 'exit', or 'q' to exit")
    print("="*70)
    
    while True:
        query = input("\n🔍 Enter your query: ").strip()
        
        # Exit conditions
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nExiting interactive search...")
            break
        
        # Skip empty queries
        if not query:
            continue
        
        # Check for category filter
        filter_category = None
        if query.lower().startswith('filter:'):
            parts = query.split(':', 1)
            if len(parts) == 2:
                filter_category = parts[1].strip()
                query_text = input("Enter query text: ").strip()
                if not query_text:
                    continue
                query = query_text
            else:
                print("Invalid filter format. Use: filter:CategoryName")
                continue
        
        # Perform search
        try:
            results = semantic_search(
                collection, 
                model, 
                query, 
                n_results=5,
                filter_category=filter_category
            )
            display_results(results, query)
        
        except Exception as e:
            print(f"\n❌ Error during search: {e}")
            continue


def save_search_statistics(collection: chromadb.Collection) -> None:
    """
    Save database statistics to file for documentation.
    
    Args:
        collection: ChromaDB collection to analyze
    """
    stats = {
        "collection_name": collection.name,
        "total_documents": collection.count(),
        "metadata": collection.metadata,
        "timestamp": datetime.now().isoformat()
    }
    
    # Create output directory
    os.makedirs("data/vector_db", exist_ok=True)
    
    # Save statistics
    stats_file = "data/vector_db/database_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"    ✓ Statistics saved to: {stats_file}")


def main():
    """
    Main execution function for Part 4: Vector Database Implementation.
    
    Steps:
        1. Load data from Parts 2 and 3
        2. Load embedding model
        3. Create and populate ChromaDB
        4. Verify persistence
        5. Display statistics
        6. Run test queries
        7. Interactive search (optional)
        
    Returns:
        chromadb.Collection: Populated and verified collection
    """
    print("=" * 70)
    print("PART 4: VECTOR DATABASE IMPLEMENTATION")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # STEP 1: Load data from previous parts
        print("Step 1: Loading data from previous parts...")
        chunks_data, embeddings = load_data()
        print(f"    ✓ Loaded {len(chunks_data)} chunks")
        print(f"    ✓ Loaded {len(embeddings)} embeddings ({embeddings.shape[1]} dimensions)")
        print()
        
        # STEP 2: Load the SAME model used in Part 3
        print("Step 2: Loading embedding model...")
        model_name = 'all-mpnet-base-v2'
        model = SentenceTransformer(model_name)
        embedding_dim = model.get_sentence_embedding_dimension()
        print(f"    ✓ Model loaded: {model_name}")
        print(f"    ✓ Embedding dimensions: {embedding_dim}")
        print()
        
        # Verify dimension match
        if embeddings.shape[1] != embedding_dim:
            raise ValueError(
                f"Dimension mismatch! Embeddings are {embeddings.shape[1]}D "
                f"but model produces {embedding_dim}D vectors!"
            )
        
        # STEP 3: Create ChromaDB collection
        print("Step 3: Creating ChromaDB collection...")
        collection = create_chromadb(chunks_data, embeddings)
        print()
        
        # STEP 4: Verify persistence
        print("Step 4: Verifying data persistence...")
        collection = verify_persistence()
        print()
        
        # STEP 5: Display database statistics
        print("Step 5: Analyzing database...")
        display_database_stats(collection)
        
        # STEP 6: Save statistics
        print("Step 6: Saving statistics...")
        save_search_statistics(collection)
        print()
        
        # STEP 7: Run test queries or interactive search
        print("Step 7: Search Interface")
        print("="*70)
        print("\nChoose mode:")
        print("  1. Run predefined test queries (5 diverse examples)")
        print("  2. Interactive search (enter your own queries)")
        print("  3. Both (test queries first, then interactive)")
        print("  4. Skip (exit)")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            run_test_queries(collection, model)
        elif choice == "2":
            interactive_search(collection, model)
        elif choice == "3":
            run_test_queries(collection, model)
            input("\nPress Enter to start interactive search...")
            interactive_search(collection, model)
        else:
            print("\nSkipping search interface...")
        
        # Completion message
        print("\n" + "="*70)
        print("✓ PART 4 COMPLETE!")
        print("="*70)
        print(f"\nSummary:")
        print(f"  • Database: ChromaDB (Persistent)")
        print(f"  • Collection: intelligent_content_retrieval")
        print(f"  • Documents: {collection.count()}")
        print(f"  • Model: {model_name} ({embedding_dim}D)")
        print(f"  • Distance metric: Cosine similarity")
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return collection
    
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Please ensure Parts 2 and 3 have been completed.")
        raise
    
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        raise
    
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()