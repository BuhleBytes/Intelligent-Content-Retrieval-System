"""
Part 5: Semantic Search Interface
Intelligent Content Retrieval System

This script provides an interactive search interface using the ChromaDB
vector database created in Part 4.

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

def typewriter(text, speed=0.03):
    """
    Print text with typewriter effect for natural chatbot feel.
    
    Args:
        text: Text to print
        speed: Delay between characters (default: 0.03s)
    """
    for char in text:
        print(char, end="", flush=True)
        time.sleep(speed)
    print()  # New line at the end


def typewriter_fast(text):
    """Fast typewriter for longer messages."""
    typewriter(text, speed=0.01)


def typewriter_slow(text):
    """Slow typewriter for emphasis."""
    typewriter(text, speed=0.05)


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def load_existing_database():
    """
    Load the existing ChromaDB collection from Part 4.
    
    Returns:
        tuple: (collection, model)
        
    Raises:
        ValueError: If collection doesn't exist
    """
    print("\n" + "="*70)
    typewriter("🤖 INITIALIZING INTELLIGENT CONTENT RETRIEVAL SYSTEM")
    print("="*70)
    
    try:
        # Connect to the SAME persistent database from Part 4
        typewriter("\n⏳ Loading ChromaDB...")
        client = chromadb.PersistentClient(path="data/chromadb")
        collection = client.get_collection(name="intelligent_content_retrieval")
        
        typewriter(f"✓ Database loaded successfully")
        print(f"  • Collection: {collection.name}")
        print(f"  • Total documents: {collection.count()}")
        
        # Load the SAME model used in Part 3 & 4
        typewriter("\n⏳ Loading AI model...")
        model_name = 'all-mpnet-base-v2'
        model = SentenceTransformer(model_name)
        
        typewriter(f"✓ Model loaded: {model_name}")
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
    """
    Perform semantic search on the existing database.
    
    Args:
        collection: ChromaDB collection from Part 4
        model: SentenceTransformer model (all-mpnet-base-v2)
        query_text: User's natural language query
        n_results: Number of results to return (default: 5)
        filter_category: Optional category filter
        
    Returns:
        dict: Search results with documents, metadata, and distances
    """
    typewriter(f"\n🔍 Searching for: '{query_text}'")
    if filter_category:
        typewriter(f"📁 Filtering by category: {filter_category}")
    
    # Generate query embedding using YOUR model (critical!)
    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True  # Must match Part 3
    )
    
    # Build query parameters
    query_params = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": n_results
    }
    
    # Add category filter if specified
    if filter_category:
        query_params["where"] = {"source_category": filter_category}
    
    # Query the database
    results = collection.query(**query_params)
    
    typewriter(f"✓ Found {len(results['documents'][0])} results\n")
    
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
)->List[Dict]:
    """
    Perform hybrid search combining semantic similarity with keyword matching.

    ALGORIITHM:
    1. Get MORE results from semantic search (cast wider net)
    2. Calculate keywprd match score for each result
    3. Combine scores: hybrid_score = (semantic_weight * semantic)+keyword
    4. Re-rank result by hybrid score
    5. Return top N result
    """
    typewriter(f"\n🔍 Searching for: '{query_text}'")
    if filter_category:
        typewriter(f"📁 Filtering by category: {filter_category}")
    
    if keywords:
        typewriter(f"🔑 Keywords: {', '.join(keywords)}")
    typewriter(f"⚙️  Using: Hybrid Search (Semantic: {semantic_weight*100:.0f}% | Keyword: {keyword_weight*100:.0f}%)")

    #STEP 1: GET SEMANTIC RESULTS 
    #Retrieve more result 2x-3x more than needed - then apply re-ranking

    retrieval_count = min(n_results*3, 50) #Get up to 50 results for re-ranking
    
    query_embedding = model.encode(
        query_text,
        convert_to_numpy = True,
        normalize_embeddings = True
    )

    query_params = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": retrieval_count
    }

    if filter_category:
        query_params["where"]  = {"source_category":filter_category}
    
    semantic_results = collection.query(**query_params)

    #If no keywords provided, return semantic results
    if not keywords or len(keywords) == 0:
        typewriter("    ⚠️No keywords provided, using semantic search only")
        return {
            'documents': [semantic_results['documents'][0][:n_results]],
            'metadatas': [semantic_results['metadatas'][0][:n_results]] ,
            'distances': [semantic_results['distances'][0][:n_results]]
        }
    
    # STEP 2: Calculate hybrid scores
    hybrid_results = []

    for doc, metadata, distance in zip(
        semantic_results['documents'][0],
        semantic_results['metadatas'][0],
        semantic_results['distances'][0]
    ):
        
        #Convert the distance to similarity
        semantic_score = 1 - distance  # ****WHY THIS IS THE CASE? AND WHY 1-distance become similarity??*****

        #Calculate keyword match score
        keyword_score = calculate_keyword_score(doc, keywords)
        
        #Combine scores using weights
        hybrid_score = (semantic_weight * semantic_score) + (keyword_weight*keyword_score)

        hybrid_results.append({
            'document': doc,
            'metadata': metadata,
            'semantic_score': semantic_score,
            'keyword_score':keyword_score,
            'hybrid_score':hybrid_score,
            'distance':distance
        }
        )

        #STEP 3: Sort by hybrid score (descending-higher is better)
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        #STEP 4: Take top N results
        top_results = hybrid_results[:n_results]
        
        typewriter(f"   ✓ Found {len(top_results)} results (re-ranked from {retrieval_count} candidates)\n")

        return top_results
    

def calculate_keyword_score(text: str, keywords: List[str])->float:
    """
    Calculate keyword match score for text chunk...

    ALGORITHM:
        - For each keyword, check if it appears in text (case-insensitive)
        - Score = (number of matched keywords)/(total keywords)
        - Returns value between 0.0 (no match) and 1.0 (all keyword match)
        
        ARGS: 
            text: Text to search in
            keyword: List of keywords  to find
        """

    if not keywords or len(keywords) == 0:
        return 0.0
        
    text_lower = text.lower()
    matches = 0
        
    for keyword in keywords:
        keyword_lower = keyword.lower().strip()
        if keyword_lower in text_lower:
            matches+=1

    return matches/len(keywords)
        

def display_results(results, query_text: str, search_mode: str = "semantic") -> None:
    """
    Display search results in a clean, readable format.
    
    Args:
        results: Results from semantic_search() or hybrid_search()
        query_text: Original query for context
        search_mode: "semantic" or "hybrid"
    """
    print("\n" + "="*70)
    if search_mode == "hybrid":
        typewriter("📊 HYBRID SEARCH RESULTS")
    else:
        typewriter("📊 SEMANTIC SEARCH RESULTS")
    print("="*70)
    print(f"Query: {query_text}")
    print("="*70)
    
    typewriter_fast("\nℹ️  Note: Results show semantically similar content chunks,")
    typewriter_fast("   ranked by relevance. Read the chunks to find your information.\n")
    
    # Handle both semantic and hybrid result formats
    if search_mode == "semantic":
        # Semantic results (from ChromaDB directly)
        if not results['documents'][0]:
            typewriter("❌ No results found!")
            typewriter("💡 Try rephrasing your query or using different keywords.\n")
            return
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            similarity = 1 - distance
            
            print(f"\n{'─'*70}")
            typewriter(f"📄 RESULT #{i}")
            print(f"{'─'*70}")
            print(f"📊 Similarity Score: {similarity:.3f} {'🔥' if similarity > 0.7 else '✓' if similarity > 0.5 else '⚠️'}")
            print(f"📁 Category: {metadata['source_category']}")
            print(f"🌐 Source: {metadata['source_url'][:60]}...")
            print(f"📄 Chunk: {metadata.get('chunk_index', '?')} of {metadata.get('total_chunks_from_source', '?')}")
            
            print(f"\n📝 Content Preview:")
            print(f"{doc[:300]}...")
            print()
    
    else:
        # Hybrid results (list of dicts)
        if not results:
            typewriter("❌ No results found!")
            typewriter("💡 Try rephrasing your query or using different keywords.\n")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n{'─'*70}")
            typewriter(f"📄 RESULT #{i}")
            print(f"{'─'*70}")
            print(f"📊 Hybrid Score: {result['hybrid_score']:.3f} {'🔥' if result['hybrid_score'] > 0.7 else '✓' if result['hybrid_score'] > 0.5 else '⚠️'}")
            print(f"   ├─ Semantic: {result['semantic_score']:.3f}")
            print(f"   └─ Keyword: {result['keyword_score']:.3f}")
            print(f"📁 Category: {result['metadata']['source_category']}")
            print(f"🌐 Source: {result['metadata']['source_url'][:60]}...")
            print(f"📄 Chunk: {result['metadata'].get('chunk_index', '?')} of {result['metadata'].get('total_chunks_from_source', '?')}")
            
            print(f"\n📝 Content Preview:")
            print(f"{result['document'][:300]}...")
            print()


# ============================================================================
# TEST MODE
# ============================================================================

def test_semantic_search(collection, model):
    """
    Test semantic search with predefined edge-case queries.
    
    Args:
        collection: ChromaDB collection
        model: SentenceTransformer model
    """
    print("\n" + "="*70)
    typewriter("🧪 RUNNING PREDEFINED TEST QUERIES (SEMANTIC SEARCH)")
    print("="*70)
    typewriter_fast("\n💡 These queries test the system's ability to understand")
    typewriter_fast("   complex concepts and find semantically similar content.\n")
    
    # Organized test queries by category
    test_queries = {
        "Machine Learning & AI": [
            "How can computers improve their performance without being explicitly programmed?",
            "What are the ethical concerns about using AI systems to make decisions about people?",
            "comparison between biological brains and machine learning systems"
        ],
        "Wildlife Conservation": [
            "What drove someone to sacrifice everything for endangered animals?",
            "How did one animal's death change conservation efforts in Africa?",
            "Can fighting against local communities actually help save endangered species?"
        ],
        "Nuclear Testing & Environment": [
            "How did Cold War politics lead to environmental health problems decades later?",
            "Which hemisphere experienced more radioactive contamination and why?",
            "What role do ocean currents and atmospheric circulation play in spreading nuclear contamination globally?"
        ],
        "TensorFlow & Optimization": [
            "When does optimization actually make code slower instead of faster?",
            "Why would the same function produce different side effects depending on how it's executed?",
            "How can I tell if my code is being recompiled too many times?"
        ]
    }
    
    query_count = 0
    for category, queries in test_queries.items():
        print(f"\n{'━'*70}")
        typewriter(f"📂 Category: {category}")
        print(f"{'━'*70}")
        
        for query in queries:
            query_count += 1
            typewriter(f"\n🔢 Test Query {query_count}/{sum(len(q) for q in test_queries.values())}")
            
            results = semantic_search(collection, model, query, n_results=3)
            display_results(results, query, search_mode="semantic")
            
            user_input = input("\n⏯️  Press Enter to continue (or 'q' to quit testing): ").strip().lower()
            if user_input == 'q':
                typewriter("\n🛑 Testing stopped by user.\n")
                return
    
    print("\n" + "="*70)
    typewriter("✅ ALL TEST QUERIES COMPLETED!")
    print("="*70 + "\n")


def test_hybrid_search(collection, model):
    """
    Test hybrid search with example queries and keywords.
    
    Args:
        collection: ChromaDB collection
        model: SentenceTransformer model
    """
    print("\n" + "="*70)
    typewriter("🧪 RUNNING HYBRID SEARCH TESTS")
    print("="*70)
    typewriter_fast("\n💡 These tests demonstrate how keywords improve precision\n")
    
    # Test cases with queries and associated keywords
    test_cases = [
        {
            "query": "How does TensorFlow optimize computational graphs?",
            "keywords": ["tensorflow", "graph", "optimization"],
            "category": "TensorFlow"
        },
        {
            "query": "What environmental damage did nuclear testing cause?",
            "keywords": ["radiation", "contamination", "radioactive"],
            "category": "Nuclear Testing"
        },
        {
            "query": "How do neural networks learn patterns?",
            "keywords": ["neural", "training", "backpropagation"],
            "category": "Machine Learning"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'━'*70}")
        typewriter(f"🔢 Test Case {i}/{len(test_cases)}: {test_case['category']}")
        print(f"{'━'*70}")
        
        results = hybrid_search(
            collection,
            model,
            test_case['query'],
            keywords=test_case['keywords'],
            n_results=3,
            semantic_weight=0.7,
            keyword_weight=0.3
        )
        
        display_results(results, test_case['query'], search_mode="hybrid")
        
        user_input = input("\n⏯️  Press Enter to continue (or 'q' to quit testing): ").strip().lower()
        if user_input == 'q':
            typewriter("\n🛑 Testing stopped by user.\n")
            return
    
    print("\n" + "="*70)
    typewriter("✅ HYBRID SEARCH TESTS COMPLETED!")
    print("="*70 + "\n")


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_search(collection, model):
    """
    Interactive search mode where users can enter custom queries.
    
    Args:
        collection: ChromaDB collection
        model: SentenceTransformer model
    """
    print("\n" + "="*70)
    typewriter("💬 INTERACTIVE SEARCH MODE")
    print("="*70)
    
    typewriter("\n🤖 I can help you find information from my knowledge base:")
    typewriter_fast("   • Machine Learning & AI concepts")
    typewriter_fast("   • Dian Fossey & Gorilla conservation")
    typewriter_fast("   • Nuclear weapons tests & environmental impacts")
    typewriter_fast("   • TensorFlow graphs & optimization")
    
    typewriter("\n💡 Tips for better results:")
    typewriter_fast("   • Ask conceptual questions (not just keywords)")
    typewriter_fast("   • Be specific about what you're looking for")
    typewriter_fast("   • Use natural language")
    
    typewriter("\n📝 Commands:")
    typewriter_fast("   • Type 'hybrid' to switch to hybrid search")
    typewriter_fast("   • Type 'semantic' to switch to semantic search")
    typewriter_fast("   • Type 'filter' to search within a specific category")
    typewriter_fast("   • Type 'config' to adjust search settings")
    typewriter_fast("   • Type 'back' to return to main menu")
    typewriter_fast("   • Type 'quit' to exit the system\n")
    
    # Default settings
    search_mode = "semantic"
    n_results = 5
    filter_category = None
    keywords = []
    semantic_weight = 0.7
    keyword_weight = 0.3
    
    while True:
        print("─"*70)
        typewriter_fast(f"\n⚙️  Current Settings: {search_mode.upper()} | Results: {n_results} | Filter: {filter_category or 'None'}")
        
        user_query = input("\n🔍 Your query: ").strip()
        
        if not user_query:
            typewriter("⚠️  Please enter a query.\n")
            continue
        
        # Handle commands
        if user_query.lower() in ['back', 'menu']:
            typewriter("\n↩️  Returning to main menu...\n")
            return
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            typewriter("\n👋 Thanks for using the Intelligent Content Retrieval System!")
            typewriter("   Goodbye!\n")
            exit(0)
        
        if user_query.lower() == 'hybrid':
            search_mode = "hybrid"
            typewriter("\n✓ Switched to Hybrid Search")
            typewriter("   You can now add keywords to improve precision!\n")
            continue
        
        if user_query.lower() == 'semantic':
            search_mode = "semantic"
            keywords = []
            typewriter("\n✓ Switched to Semantic Search\n")
            continue
        
        if user_query.lower() == 'config':
            typewriter("\n⚙️  CONFIGURATION")
            typewriter_fast("   Current settings:")
            print(f"   • Search Mode: {search_mode}")
            print(f"   • Number of Results: {n_results}")
            print(f"   • Category Filter: {filter_category or 'None'}")
            if search_mode == "hybrid":
                print(f"   • Keywords: {keywords if keywords else 'None'}")
                print(f"   • Semantic Weight: {semantic_weight*100:.0f}%")
                print(f"   • Keyword Weight: {keyword_weight*100:.0f}%")
            
            config_input = input("\n   What would you like to change? (results/filter/weights/back): ").strip().lower()
            
            if config_input == 'results':
                try:
                    new_count = int(input("   Enter number of results (1-20): "))
                    if 1 <= new_count <= 20:
                        n_results = new_count
                        typewriter(f"\n   ✓ Results set to {n_results}")
                    else:
                        typewriter("\n   ⚠️  Please enter a number between 1 and 20")
                except:
                    typewriter("\n   ⚠️  Invalid input")
            
            elif config_input == 'filter':
                typewriter("\n   📂 Available categories:")
                print("      1. All Categories (remove filter)")
                print("      2. News")
                print("      3. Educational")
                print("      4. Technical Documentation")
                print("      5. Research Publication")
                
                filter_input = input("\n      Select category (1-5): ").strip()
                filter_map = {
                    '1': None,
                    '2': 'News',
                    '3': 'Educational',
                    '4': 'Technical Documentation',
                    '5': 'Research Publication'
                }
                
                if filter_input in filter_map:
                    filter_category = filter_map[filter_input]
                    typewriter(f"\n   ✓ Filter set to: {filter_category or 'None'}")
            
            elif config_input == 'weights' and search_mode == "hybrid":
                typewriter("\n   ⚖️  Adjust search balance:")
                typewriter_fast("      Current: Semantic {:.0f}% | Keyword {:.0f}%".format(
                    semantic_weight*100, keyword_weight*100
                ))
                
                try:
                    sem_input = int(input("\n      Semantic weight % (0-100): "))
                    if 0 <= sem_input <= 100:
                        semantic_weight = sem_input / 100
                        keyword_weight = 1 - semantic_weight
                        typewriter(f"\n   ✓ Weights: Semantic {semantic_weight*100:.0f}% | Keyword {keyword_weight*100:.0f}%")
                    else:
                        typewriter("\n   ⚠️  Please enter a number between 0 and 100")
                except:
                    typewriter("\n   ⚠️  Invalid input")
            
            print()
            continue
        
        if user_query.lower() == 'filter':
            typewriter("\n📂 Available categories:")
            print("   1. News")
            print("   2. Educational")
            print("   3. Technical Documentation")
            print("   4. Research Publication")
            
            category_choice = input("\n   Select category (1-4) or Enter to skip: ").strip()
            
            category_map = {
                '1': 'News',
                '2': 'Educational',
                '3': 'Technical Documentation',
                '4': 'Research Publication'
            }
            
            filter_category = category_map.get(category_choice)
            
            user_query = input("\n🔍 Your query: ").strip()
            if not user_query:
                continue
        
        # Ask for keywords if in hybrid mode
        if search_mode == "hybrid":
            keyword_input = input("🔑 Add keywords (space-separated, or Enter to skip): ").strip()
            if keyword_input:
                keywords = [kw.strip() for kw in keyword_input.split()]
            else:
                keywords = []
        
        # Perform search based on mode
        if search_mode == "semantic":
            results = semantic_search(
                collection, model, user_query, 
                n_results=n_results, 
                filter_category=filter_category
            )
            display_results(results, user_query, search_mode="semantic")
        else:
            results = hybrid_search(
                collection, model, user_query,
                keywords=keywords,
                n_results=n_results,
                filter_category=filter_category,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )
            display_results(results, user_query, search_mode="hybrid")
        
        print("\n" + "─"*70)


# ============================================================================
# UI FUNCTIONS
# ============================================================================

def show_welcome_message():
    """Display welcome message and system information."""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    typewriter_slow("║" + "  🤖  INTELLIGENT CONTENT RETRIEVAL SYSTEM  🤖  ".center(68) + "║")
    print("║" + " "*68 + "║")
    typewriter_fast("║" + "  Semantic Search Powered by AI".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    print()


def show_main_menu():
    """Display main menu options."""
    print("\n" + "═"*70)
    typewriter("📋 MAIN MENU")
    print("═"*70)
    typewriter("\n🤖 What would you like to do?\n")
    
    print("  1️⃣  Run semantic search tests")
    typewriter_fast("      (Test pure AI-powered semantic search)")
    print()
    print("  2️⃣  Run hybrid search tests")
    typewriter_fast("      (Test semantic + keyword combination)")
    print()
    print("  3️⃣  Interactive search")
    typewriter_fast("      (Ask your own questions with full configuration)")
    print()
    print("  4️⃣  View system information")
    typewriter_fast("      (See what content is available for search)")
    print()
    print("  5️⃣  Exit")
    typewriter_fast("      (Close the application)")
    print("\n" + "─"*70)


def show_system_info(collection):
    """
    Display information about the system's knowledge base.
    
    Args:
        collection: ChromaDB collection
    """
    print("\n" + "="*70)
    typewriter("ℹ️  SYSTEM INFORMATION")
    print("="*70)
    
    typewriter(f"\n📊 Database Statistics:")
    print(f"   • Total chunks: {collection.count()}")
    print(f"   • Embedding model: all-mpnet-base-v2 (768 dimensions)")
    print(f"   • Distance metric: Cosine similarity")
    
    typewriter(f"\n🔍 Search Capabilities:")
    print(f"   • Semantic Search: AI-powered conceptual matching")
    print(f"   • Hybrid Search: Semantic + keyword filtering")
    print(f"   • Category Filtering: Search within specific domains")
    print(f"   • Adjustable Results: 1-20 results per query")
    
    typewriter(f"\n📚 Knowledge Base Content:")
    
    # Sample a few documents to show categories
    sample = collection.peek(limit=20)
    categories = {}
    
    for metadata in sample['metadatas']:
        category = metadata.get('source_category', 'Unknown')
        if category not in categories:
            categories[category] = {
                'url': metadata.get('source_url', 'N/A'),
                'count': 0
            }
        categories[category]['count'] += 1
    
    for category, info in categories.items():
        print(f"\n   📁 {category}")
        print(f"      Source: {info['url'][:60]}...")
        print(f"      Chunks: ~{info['count']} (from sample)")
    
    print("\n" + "="*70 + "\n")
    input("Press Enter to return to main menu...")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main execution function for Part 5: Semantic Search Interface.
    """
    try:
        # Show welcome message
        show_welcome_message()
        
        # Load database and model
        collection, model = load_existing_database()
        
        typewriter("\n✅ System ready!\n")
        time.sleep(0.5)
        
        # Main interaction loop
        while True:
            show_main_menu()
            
            user_choice = input("\n👉 Enter your choice (1-5): ").strip()
            
            if user_choice == '1':
                # Run semantic search tests
                test_semantic_search(collection, model)
                
            elif user_choice == '2':
                # Run hybrid search tests
                test_hybrid_search(collection, model)
                
            elif user_choice == '3':
                # Interactive search mode
                interactive_search(collection, model)
                
            elif user_choice == '4':
                # Show system information
                show_system_info(collection)
                
            elif user_choice in ['5', 'quit', 'exit', 'q']:
                # Exit
                print("\n" + "═"*70)
                typewriter("👋 Thanks for using the Intelligent Content Retrieval System!")
                typewriter("   Author: Buhle Mlandu")
                typewriter("   Part 5: Semantic Search Interface (with Hybrid Search)")
                print("═"*70 + "\n")
                break
                
            else:
                typewriter("\n⚠️  Invalid choice. Please enter 1, 2, 3, 4, or 5.\n")
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user.")
        typewriter("👋 Goodbye!\n")
    
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        print("Please ensure Parts 1-4 have been completed successfully.\n")
        raise


if __name__ == "__main__":
    main()