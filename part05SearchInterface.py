"""
Part 5: Semantic Search Interface (CLEANED UP VERSION)
Intelligent Content Retrieval System

Author: Buhle Mlandu
"""

import chromadb
import time
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import numpy as np


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def typewriter(text, speed=0.02):
    """Print text with typewriter effect - FASTER for less clutter."""
    for char in text:
        print(char, end="", flush=True)
        time.sleep(speed)
    print()


def typewriter_fast(text):
    """Fast typewriter for longer messages."""
    typewriter(text, speed=0.005)


def typewriter_slow(text):
    """Slow typewriter for emphasis."""
    typewriter(text, speed=0.04)


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def load_existing_database():
    """Load the existing ChromaDB collection from Part 4."""
    print("\n" + "="*70)
    typewriter("🤖 INITIALIZING INTELLIGENT CONTENT RETRIEVAL SYSTEM")
    print("="*70)
    
    try:
        typewriter("\n⏳ Loading ChromaDB...")
        client = chromadb.PersistentClient(path="data/chromadb")
        collection = client.get_collection(name="intelligent_content_retrieval")
        
        print("✓ Database loaded successfully")
        print(f"  • Collection: {collection.name}")
        print(f"  • Total documents: {collection.count()}")
        
        typewriter("\n⏳ Loading AI model...")
        model_name = 'all-mpnet-base-v2'
        model = SentenceTransformer(model_name)
        
        print(f"✓ Model loaded: {model_name}")
        print(f"  • Embedding dimensions: {model.get_sentence_embedding_dimension()}")
        
        return collection, model
    
    except ValueError as e:
        raise ValueError(
            f"Collection 'intelligent_content_retrieval' not found!\n"
            f"Please run Part 4 first to create the database.\n"
            f"Error: {e}"
        )
    except Exception as e:
        raise Exception(f"Error loading database: {e}")


# ============================================================================
# SEARCH FUNCTIONS
# ============================================================================

def semantic_search(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    query_text: str,
    n_results: int = 5,
    filter_category: str = None
) -> Dict:
    """Perform semantic search on the existing database."""
    print(f"\n🔍 Searching: '{query_text}'")
    if filter_category:
        print(f"📁 Filter: {filter_category}")
    print("⚙️  Mode: Semantic Search")
    
    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    query_params = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": n_results
    }
    
    if filter_category:
        query_params["where"] = {"source_category": filter_category}
    
    results = collection.query(**query_params)
    
    print(f"✓ Found {len(results['documents'][0])} results\n")
    
    return results


def hybrid_search(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    query_text: str,
    keywords: List[str] = None,
    n_results: int = 5,
    filter_category: str = None,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[Dict]:
    """Perform hybrid search combining semantic similarity with keyword matching."""
    print(f"\n🔍 Searching: '{query_text}'")
    if filter_category:
        print(f"📁 Filter: {filter_category}")
    if keywords:
        print(f"🔑 Keywords: {', '.join(keywords)}")
    print(f"⚙️  Mode: Hybrid Search ({semantic_weight*100:.0f}% semantic + {keyword_weight*100:.0f}% keyword)")
    
    # STEP 1: Get MORE semantic results for re-ranking
    # FIX: Increase retrieval to ensure we get enough results AFTER keyword filtering
    retrieval_count = min(n_results * 10, 100)  # Changed from 3x to 10x, max 100
    
    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    query_params = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": retrieval_count
    }
    
    if filter_category:
        query_params["where"] = {"source_category": filter_category}
    
    semantic_results = collection.query(**query_params)
    
    # If no keywords, return semantic results
    if not keywords or len(keywords) == 0:
        print("⚠️  No keywords - using semantic search only")
        return {
            'documents': [semantic_results['documents'][0][:n_results]],
            'metadatas': [semantic_results['metadatas'][0][:n_results]],
            'distances': [semantic_results['distances'][0][:n_results]]
        }
    
    # STEP 2: Calculate hybrid scores
    hybrid_results = []
    
    for doc, metadata, distance in zip(
        semantic_results['documents'][0],
        semantic_results['metadatas'][0],
        semantic_results['distances'][0]
    ):
        semantic_score = 1 - distance
        keyword_score = calculate_keyword_score(doc, keywords)
        
        # FIX: Don't filter out results with low keyword scores
        # Just let the hybrid score naturally rank them lower
        hybrid_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
        
        hybrid_results.append({
            'document': doc,
            'metadata': metadata,
            'semantic_score': semantic_score,
            'keyword_score': keyword_score,
            'hybrid_score': hybrid_score,
            'distance': distance
        })
    
    # STEP 3: Sort by hybrid score (higher = better)
    hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    # STEP 4: Return top N results
    top_results = hybrid_results[:n_results]
    
    print(f"✓ Found {len(top_results)} results (from {retrieval_count} candidates)\n")
    
    return top_results


def calculate_keyword_score(text: str, keywords: List[str]) -> float:
    """
    Calculate keyword match score (proportion of keywords found).
    
    Returns value between 0.0 and 1.0
    """
    if not keywords or len(keywords) == 0:
        return 0.0
    
    text_lower = text.lower()
    matches = 0
    
    for keyword in keywords:
        keyword_lower = keyword.lower().strip()
        if keyword_lower in text_lower:
            matches += 1
    
    return matches / len(keywords)


def display_results(results, query_text: str, search_mode: str = "semantic") -> None:
    """Display search results in a CLEAN, readable format."""
    print("\n" + "="*70)
    print(f"{'HYBRID' if search_mode == 'hybrid' else 'SEMANTIC'} SEARCH RESULTS")
    print("="*70)
    print(f"Query: {query_text}\n")
    
    # Handle both result formats
    if search_mode == "semantic":
        if not results['documents'][0]:
            print("❌ No results found!")
            print("💡 Try rephrasing your query or removing filters.\n")
            return
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            similarity = 1 - distance
            
            # Clean, compact display
            print(f"\n📄 Result #{i}  |  Similarity: {similarity:.3f} {'🔥' if similarity > 0.7 else '✓' if similarity > 0.5 else '⚠️'}")
            print(f"   {metadata['source_category']} | Chunk {metadata.get('chunk_index', '?')}/{metadata.get('total_chunks_from_source', '?')}")
            print(f"   {metadata['source_url'][:65]}...")
            print(f"\n   {doc}...\n")  # FIX: Limit to 250 chars
    
    else:
        # Hybrid results
        if not results:
            print("❌ No results found!")
            print("💡 Try different keywords or rephrasing.\n")
            return
        
        for i, result in enumerate(results, 1):
            # Clean, compact display
            print(f"\n📄 Result #{i}  |  Hybrid: {result['hybrid_score']:.3f} {'🔥' if result['hybrid_score'] > 0.7 else '✓' if result['hybrid_score'] > 0.5 else '⚠️'}")
            print(f"   Semantic: {result['semantic_score']:.3f} | Keywords: {result['keyword_score']:.3f}")
            print(f"   {result['metadata']['source_category']} | Chunk {result['metadata'].get('chunk_index', '?')}/{result['metadata'].get('total_chunks_from_source', '?')}")
            print(f"   {result['metadata']['source_url'][:65]}...")
            print(f"\n   {result['document'][:250]}...\n")  # FIX: Limit to 250 chars


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_search(collection, model):
    """Interactive search mode with clean UX."""
    print("\n" + "="*70)
    typewriter("💬 INTERACTIVE SEARCH MODE")
    print("="*70)
    
    typewriter("\n🤖 I can search across:")
    print("   • Machine Learning & AI")
    print("   • Gorilla conservation")
    print("   • Nuclear testing impacts")
    print("   • TensorFlow optimization")
    
    typewriter("\n📝 Commands:")
    print("   'hybrid' - Switch to hybrid search")
    print("   'semantic' - Switch to semantic search")
    print("   'config' - Adjust settings")
    print("   'back' - Return to menu")
    print("   'quit' - Exit\n")
    
    # Default settings
    search_mode = "semantic"
    n_results = 5
    filter_category = None
    keywords = []
    semantic_weight = 0.7
    keyword_weight = 0.3
    
    while True:
        print("─"*70)
        print(f"\n⚙️  {search_mode.upper()} | {n_results} results | Filter: {filter_category or 'None'}")
        
        user_query = input("\n🔍 Your query: ").strip()
        
        if not user_query:
            print("⚠️  Please enter a query.\n")
            continue
        
        # Handle commands
        if user_query.lower() in ['back', 'menu']:
            print("\n↩️  Returning to main menu...\n")
            return
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!\n")
            exit(0)
        
        if user_query.lower() == 'hybrid':
            search_mode = "hybrid"
            print("\n✓ Switched to Hybrid Search\n")
            continue
        
        if user_query.lower() == 'semantic':
            search_mode = "semantic"
            keywords = []
            print("\n✓ Switched to Semantic Search\n")
            continue
        
        if user_query.lower() == 'config':
            print("\n⚙️  CONFIGURATION")
            print(f"   Mode: {search_mode} | Results: {n_results} | Filter: {filter_category or 'None'}")
            
            if search_mode == "hybrid":
                print(f"   Weights: {semantic_weight*100:.0f}% semantic, {keyword_weight*100:.0f}% keyword")
            
            config_choice = input("\n   Change: (results/filter/weights/back): ").strip().lower()
            
            if config_choice == 'results':
                try:
                    new_n = int(input("   Results (1-20): "))
                    if 1 <= new_n <= 20:
                        n_results = new_n
                        print(f"\n   ✓ Set to {n_results} results")
                except:
                    print("\n   ⚠️  Invalid")
            
            elif config_choice == 'filter':
                print("\n   1. All  2. News  3. Educational  4. Technical  5. Research")
                f_choice = input("   Select (1-5): ").strip()
                filter_map = {'1': None, '2': 'News', '3': 'Educational', 
                             '4': 'Technical Documentation', '5': 'Research Publication'}
                filter_category = filter_map.get(f_choice)
                print(f"\n   ✓ Filter: {filter_category or 'None'}")
            
            elif config_choice == 'weights' and search_mode == "hybrid":
                try:
                    sem = int(input("   Semantic % (0-100): "))
                    if 0 <= sem <= 100:
                        semantic_weight = sem / 100
                        keyword_weight = 1 - semantic_weight
                        print(f"\n   ✓ {semantic_weight*100:.0f}% semantic, {keyword_weight*100:.0f}% keyword")
                except:
                    print("\n   ⚠️  Invalid")
            
            print()
            continue
        
        # Ask for keywords if hybrid mode
        if search_mode == "hybrid":
            kw_input = input("🔑 Keywords (space-separated, Enter=none): ").strip()
            keywords = [k.strip() for k in kw_input.split()] if kw_input else []
        
        # Perform search
        if search_mode == "semantic":
            results = semantic_search(collection, model, user_query, n_results, filter_category)
            display_results(results, user_query, "semantic")
        else:
            results = hybrid_search(collection, model, user_query, keywords, n_results, 
                                   filter_category, semantic_weight, keyword_weight)
            display_results(results, user_query, "hybrid")


# ============================================================================
# UI FUNCTIONS
# ============================================================================

def show_welcome_message():
    """Display welcome screen."""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    typewriter_slow("║" + "  🤖  INTELLIGENT CONTENT RETRIEVAL SYSTEM  🤖  ".center(68) + "║")
    print("║" + " "*68 + "║")
    print("║" + "  Semantic Search Powered by AI".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")


def show_main_menu():
    """Display main menu."""
    print("\n" + "="*70)
    typewriter("📋 MAIN MENU")
    print("="*70)
    print("\n  1️⃣  Interactive Search")
    print("  2️⃣  View System Info")
    print("  3️⃣  Exit\n")
    print("─"*70)


def show_system_info(collection):
    """Display system information."""
    print("\n" + "="*70)
    print("ℹ️  SYSTEM INFORMATION")
    print("="*70)
    
    print(f"\n📊 Database:")
    print(f"   • {collection.count()} chunks")
    print(f"   • Model: all-mpnet-base-v2 (768D)")
    print(f"   • Metric: Cosine similarity")
    
    print(f"\n🔍 Search Modes:")
    print(f"   • Semantic: AI-powered concept matching")
    print(f"   • Hybrid: Semantic + keyword filtering")
    
    print(f"\n📚 Content:")
    
    sample = collection.peek(limit=20)
    categories = {}
    
    for metadata in sample['metadatas']:
        cat = metadata.get('source_category', 'Unknown')
        if cat not in categories:
            categories[cat] = {'url': metadata.get('source_url', 'N/A'), 'count': 0}
        categories[cat]['count'] += 1
    
    for cat, info in categories.items():
        print(f"\n   📁 {cat}")
        print(f"      {info['url'][:60]}...")
    
    print("\n" + "="*70 + "\n")
    input("Press Enter to continue...")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    try:
        show_welcome_message()
        collection, model = load_existing_database()
        
        print("\n✅ System ready!\n")
        time.sleep(0.5)
        
        while True:
            show_main_menu()
            choice = input("\n👉 Choice (1-3): ").strip()
            
            if choice == '1':
                interactive_search(collection, model)
            elif choice == '2':
                show_system_info(collection)
            elif choice in ['3', 'quit', 'exit', 'q']:
                print("\n" + "="*70)
                print("👋 Thanks for using the Intelligent Content Retrieval System!")
                print("   Author: Buhle Mlandu")
                print("="*70 + "\n")
                break
            else:
                print("\n⚠️  Invalid choice\n")
                time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}\n")
        raise


if __name__ == "__main__":
    main()