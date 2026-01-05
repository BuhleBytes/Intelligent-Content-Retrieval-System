"""
Part 5: Semantic Search Interface
Intelligent Content Retrieval System

This script provides an interactive search interface using the ChromaDB
vector database created in Part 4.

Author: Buhle Mlandu
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import numpy as np


def load_existing_database():
    """
    Load the existing chromaDB collection from Part 4
    Returns:
       tuples (collection, model)
    """
    print("Loading the chromaDB")
    try:
        #Connect to the SAME persistent database from Part 4
        client= chromadb.PersistentClient(path="data/chromadb")
        collection = client.get_collection(name="intelligent_content_retrieval")
        
        print(f"    ✓ Database loaded successfully")
        print(f"    Collection: {collection.name}")
        print(f"    Total documents: {collection.count()}")

        model_name = 'all-mpnet-base-v2'
        model  = SentenceTransformer(model_name)

        print(f"    ✓ Model loaded: {model_name}")
        print(f"    Embedding dimensions: {model.get_sentence_embedding_dimension()}")
    
    except ValueError as e:
        raise ValueError(
            f"Collection 'intelligent_content_retrieval' not found!\n"
            f"Please run Part 4 first to create the database.\n"
            f"Error: {e}"
        )
    
    except Exception as e:
        raise Exception(f"Error loading database: {e}")

def semantic_search(
        collection: chromadb.Collection,
        model: SentenceTransformer,
        query_text: str,
        n_results: int = 5,
        filter_category: str = None
)->Dict:
    """
    Performs semantic search on the existing database.
    
    Args:
        collection: ChromaDB collection from Part 4
        model: SentenceTransformer model (all-mpnet-base-v2)
        query_text: User's natural language query
        n_result: Number of results to return (default: 5)
        filter_category: Optional category
    
    Returns:
        dict: Search results with documents, metadata and distance
    """

    print(f"\n🔍 Searching for: '{query_text}'")
    if filter_category:
        print(f"📁 Filtering by category: {filter_category}")

    # Generate query embeddings usign the user's query and the model
    query_embeddings = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True #Must match Part 3
    )

    #Build query parameters
    query_params = {
        "query_embeddings": [query_embeddings.tolist()],
        "n_results": n_results
    }

    #Add category filter if specified
    if filter_category:
        query_params["where"] = {"source_category":filter_category}
        
    results = collection.query(**query_params)
    print(f"    ✓ Found {len(results['documents'][0])} results")
    
    return results

def display_result(results:Dict, query_text:str)->None:
    """
    Display search results in a clean, readable format

    Args:
        results: Results from semantic_searh()
        query_text: Original query for context
    """
    print(f"\n{'='*70}")
    print(f"SEARCH RESULTS FOR: {query_text}")
    print(f"{'='*70}\n")

    if not results['documents'][0]:
        print("❌ No results found!")
        return
    
    for i, (doc, metadata, distance) in enumerate(
        zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distance'][0]
        ), 1):

        similarity = 1 - distance
        print(f"--- Result {i} ---")
        print(f"📊 Similarity Score: {similarity:.3f}")
        print(f"📁 Category: {metadata['source_category']}")
        print(f"🌐 Source: {metadata['source_url']}")
        print(f"📄 Chunk: {metadata.get('chunk_index', '?')} / {metadata.get('total_chunks_from_source', '?')}")
        print(f"\n📝 Text Preview:")
        print(f"{doc[:250]}...")
        print()


def test_semantic_search(collection, model):
    """
    Test semantic search with example queries.
    Args:
        collection: ChromaDB collection
        model: SentenceTransformer model
    """
    print("\n" + "="*70)
    print("TESTING SEMANTIC SEARCH")
    print("="*70)
    
    # <<<Test queries>>>
    #  3 time AI based

    test_queries = ["How can computers improve their performance without being explicitly programmed?",
                    "What are the ethical concerns about using AI systems to make decisions about people?",
                    "comparison between biological brains and machine learning systems",

                    "What drove someone to sacrifice everything for endangered animals?",
                    "How did one animal's death change conservation efforts in Africa?",
                    "Can fighting against local communities actually help save endangered species?",

                    "How did Cold War politics lead to environmental health problems decades later?",
                    "Which hemisphere experienced more radioactive contamination and why?",
                    "What role do ocean currents and atmospheric circulation play in spreading nuclear contamination globally?",
                    "What percentage of total nuclear tests were conducted by the United States and Soviet Union combined?",

                    "When does optimization actually make code slower instead of faster?",
                    "Why would the same function produce different side effects depending on how it's executed?",
                    "How can I tell if my code is being recompiled too many times?",
                    "What's the difference between writing instructions and following instructions in TensorFlow?"
                    ]
    
    for query in test_queries:
        results = semantic_search(collection, model, query, n_results=5)
        display_result(results, query)
        input("\nPress Enter for next query....\n")


def main():
    """Main execution for oart 5"""
    
