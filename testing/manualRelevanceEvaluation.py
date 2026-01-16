import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from part05SearchInterface import load_existing_database, semantic_search, hybrid_search

# Assumes you have these imported
# from part05SearchInterface import load_existing_database, semantic_search, hybrid_search

# ============================================================================
# STEP 1: DEFINE TEST QUERIES
# ============================================================================

test_queries = [
    # Simple factual queries
    "What is Machine Learning?",
    "What is a TensorFlow computation graph (tf.Graph)",
    "What is the Non-Proliferation Treaty?",
    "What are gorillas ?",
    
    # Domain specific queries
    "What are gorillas conservative efforts",
    "Who was Dian Fossey?",
    "Effects of nuclear testing on environment",
    "Radioactive contamination from nuclear weapons",
    "What is AutoGraph in TensorFlow?",
    "What is a Bayesian Network?",
    
    # How-to queries 
    "How to train machine learning model?",
    "How do I convert a Python function into a TensorFlow graph using tf.function?",
    
    # Application  
    "Application of machine learning",
    "Application of Nuclear Weapons"
]

# For each query, define keywords for hybrid search
query_keywords = {
    # Simple factual queries
    "What is Machine Learning?": ["machine", "learning"],
    "What is a TensorFlow computation graph (tf.Graph)": ["TensorFlow", "computation", "graph", "tf.Graph"],
    "What is the Non-Proliferation Treaty?": ["Non-Proliferation", "Treaty", "nuclear"],
    "What are gorillas ?": ["gorillas"],
    
    # Domain specific queries
    "What are gorillas conservative efforts": ["gorillas", "conservation", "efforts"],
    "Who was Dian Fossey?": ["Dian", "Fossey", "gorilla"],
    "Effects of nuclear testing on environment": ["nuclear", "testing", "environment", "effects"],
    "Radioactive contamination from nuclear weapons": ["radioactive", "contamination", "nuclear", "weapons"],
    "What is AutoGraph in TensorFlow?": ["AutoGraph", "TensorFlow"],
    "What is a Bayesian Network?": ["Bayesian", "Network", "machine", "learning"],
    
    # How-to queries
    "How to train machine learning model?": ["train", "machine", "learning", "model"],
    "How do I convert a Python function into a TensorFlow graph using tf.function?": ["convert", "Python", "function", "TensorFlow", "graph", "tf.function"],
    
    # Application
    "Application of machine learning": ["application", "machine", "learning"],
    "Application of Nuclear Weapons": ["application", "nuclear", "weapons"]
}
print(f"Total test queries: {len(test_queries)}")
print(f"Expected evaluations per mode: {len(test_queries) * 5} results")
print(f"Total evaluations (both modes): {len(test_queries) * 5 * 2} results\n")

# ============================================================================
# STEP 2: RUN QUERIES AND COLLECT MANUAL RATINGS (BOTH MODES)
# ============================================================================

def run_evaluation_comparison(collection, model, test_queries, query_keywords, results_per_query=5):
    """
    Run all test queries with BOTH semantic and hybrid search.
    Collect manual ratings for comparison.
    
    Returns:
        DataFrame with all evaluation data including search mode
    """
    evaluation_data = []
    
    print("="*70)
    print("MANUAL RELEVANCE EVALUATION - SEMANTIC vs HYBRID COMPARISON")
    print("="*70)
    print("\nRating Scale:")
    print("  0 = Not Relevant (unrelated to query)")
    print("  1 = Somewhat Relevant (tangentially related)")
    print("  2 = Highly Relevant (directly answers query)")
    print("\n" + "="*70 + "\n")
    
    for query_num, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"QUERY {query_num}/{len(test_queries)}: {query}")
        print(f"{'='*70}\n")
        
        keywords = query_keywords.get(query, [])
        
        # ====================================================================
        # MODE 1: SEMANTIC SEARCH
        # ====================================================================
        
        print(f"\n{'─'*70}")
        print(f"MODE 1: SEMANTIC SEARCH")
        print(f"{'─'*70}\n")
        
        semantic_results = semantic_search(collection, model, query, n_results=results_per_query)
        
        if not semantic_results['documents'][0]:
            print("⚠️  No results found for semantic search!")
        else:
            for i, (doc, metadata, distance) in enumerate(zip(
                semantic_results['documents'][0],
                semantic_results['metadatas'][0],
                semantic_results['distances'][0]
            ), 1):
                similarity = 1 - distance
                
                print(f"\n--- Semantic Result #{i} ---")
                print(f"Similarity Score: {similarity:.3f}")
                print(f"Category: {metadata['source_category']}")
                print(f"Source: {metadata['source_url'][:60]}...")
                print(f"\nText Preview:")
                print(f"{doc}...")
                print(f"\n{'-'*70}")
                
                # Get manual rating
                while True:
                    try:
                        rating = int(input(f"Rate SEMANTIC result #{i} (0/1/2): "))
                        if rating in [0, 1, 2]:
                            break
                        else:
                            print("Please enter 0, 1, or 2")
                    except ValueError:
                        print("Please enter a number (0, 1, or 2)")
                
                evaluation_data.append({
                    'query_num': query_num,
                    'query': query,
                    'search_mode': 'semantic',
                    'rank': i,
                    'rating': rating,
                    'similarity_score': similarity,
                    'category': metadata['source_category'],
                    'source_url': metadata['source_url'],
                    'chunk_id': metadata.get('chunk_id', 'N/A'),
                    'text_preview': doc
                })
                
                print(f"✓ Recorded rating: {rating}\n")
        
        # ====================================================================
        # MODE 2: HYBRID SEARCH
        # ====================================================================
        
        print(f"\n{'─'*70}")
        print(f"MODE 2: HYBRID SEARCH")
        print(f"Keywords: {', '.join(keywords) if keywords else 'None'}")
        print(f"{'─'*70}\n")
        
        hybrid_results = hybrid_search(
            collection, model, query, 
            keywords=keywords,
            n_results=results_per_query,
            semantic_weight=0.7,
            keyword_weight=0.3
        )
        
        if not hybrid_results:
            print("⚠️  No results found for hybrid search!")
        else:
            for i, result in enumerate(hybrid_results, 1):
                print(f"\n--- Hybrid Result #{i} ---")
                print(f"Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"  (Semantic: {result['semantic_score']:.3f} | Keyword: {result['keyword_score']:.3f})")
                print(f"Category: {result['metadata']['source_category']}")
                print(f"Source: {result['metadata']['source_url'][:60]}...")
                print(f"\nText Preview:")
                print(f"{result['document']}...")
                print(f"\n{'-'*70}")
                
                # Get manual rating
                while True:
                    try:
                        rating = int(input(f"Rate HYBRID result #{i} (0/1/2): "))
                        if rating in [0, 1, 2]:
                            break
                        else:
                            print("Please enter 0, 1, or 2")
                    except ValueError:
                        print("Please enter a number (0, 1, or 2)")
                
                evaluation_data.append({
                    'query_num': query_num,
                    'query': query,
                    'search_mode': 'hybrid',
                    'rank': i,
                    'rating': rating,
                    'similarity_score': result['semantic_score'],
                    'hybrid_score': result['hybrid_score'],
                    'keyword_score': result['keyword_score'],
                    'category': result['metadata']['source_category'],
                    'source_url': result['metadata']['source_url'],
                    'chunk_id': result['metadata'].get('chunk_id', 'N/A'),
                    'text_preview': result['document']
                })
                
                print(f"✓ Recorded rating: {rating}\n")
        
        # Progress indicator
        remaining = len(test_queries) - query_num
        print(f"\n{'='*70}")
        print(f"Progress: {query_num}/{len(test_queries)} queries completed")
        print(f"Remaining: {remaining} queries ({remaining * results_per_query * 2} results)")
        print(f"{'='*70}\n")
        
        # Optional: pause between queries
        if query_num < len(test_queries):
            input("Press Enter to continue to next query...\n")
    
    return pd.DataFrame(evaluation_data)


# ============================================================================
# STEP 3: CALCULATE METRICS (BY MODE)
# ============================================================================

def calculate_metrics_by_mode(df):
    """Calculate metrics separately for each search mode."""
    
    metrics = {}
    
    for mode in ['semantic', 'hybrid']:
        mode_df = df[df['search_mode'] == mode]
        
        if len(mode_df) == 0:
            continue
        
        mode_metrics = {}
        
        # Basic counts
        mode_metrics['total_queries'] = mode_df['query_num'].nunique()
        mode_metrics['total_results'] = len(mode_df)
        
        # Precision@5
        relevant_count = (mode_df['rating'] >= 1).sum()
        mode_metrics['precision_at_5'] = relevant_count / len(mode_df)
        
        # Mean relevance rating
        mode_metrics['mean_rating'] = mode_df['rating'].mean()
        
        # Distribution
        mode_metrics['highly_relevant_pct'] = (mode_df['rating'] == 2).sum() / len(mode_df)
        mode_metrics['somewhat_relevant_pct'] = (mode_df['rating'] == 1).sum() / len(mode_df)
        mode_metrics['not_relevant_pct'] = (mode_df['rating'] == 0).sum() / len(mode_df)
        
        # Rating by rank
        mode_metrics['rating_by_rank'] = mode_df.groupby('rank')['rating'].mean().to_dict()
        
        # Category performance
        mode_metrics['category_performance'] = mode_df.groupby('category')['rating'].mean().to_dict()
        
        metrics[mode] = mode_metrics
    
    # Calculate improvement
    if 'semantic' in metrics and 'hybrid' in metrics:
        metrics['comparison'] = {
            'precision_improvement': metrics['hybrid']['precision_at_5'] - metrics['semantic']['precision_at_5'],
            'rating_improvement': metrics['hybrid']['mean_rating'] - metrics['semantic']['mean_rating'],
        }
    
    return metrics


def print_metrics_comparison(metrics):
    """Print comparative metrics."""
    
    print("\n" + "="*70)
    print("EVALUATION METRICS - SEMANTIC vs HYBRID COMPARISON")
    print("="*70)
    
    # Semantic metrics
    if 'semantic' in metrics:
        print("\n" + "─"*70)
        print("SEMANTIC SEARCH RESULTS")
        print("─"*70)
        sem = metrics['semantic']
        print(f"\nTotal Queries: {sem['total_queries']}")
        print(f"Total Results: {sem['total_results']}")
        print(f"Precision@5: {sem['precision_at_5']:.1%}")
        print(f"Mean Rating: {sem['mean_rating']:.2f} / 2.0")
        print(f"\nHighly Relevant:   {sem['highly_relevant_pct']:.1%}")
        print(f"Somewhat Relevant: {sem['somewhat_relevant_pct']:.1%}")
        print(f"Not Relevant:      {sem['not_relevant_pct']:.1%}")
    
    # Hybrid metrics
    if 'hybrid' in metrics:
        print("\n" + "─"*70)
        print("HYBRID SEARCH RESULTS")
        print("─"*70)
        hyb = metrics['hybrid']
        print(f"\nTotal Queries: {hyb['total_queries']}")
        print(f"Total Results: {hyb['total_results']}")
        print(f"Precision@5: {hyb['precision_at_5']:.1%}")
        print(f"Mean Rating: {hyb['mean_rating']:.2f} / 2.0")
        print(f"\nHighly Relevant:   {hyb['highly_relevant_pct']:.1%}")
        print(f"Somewhat Relevant: {hyb['somewhat_relevant_pct']:.1%}")
        print(f"Not Relevant:      {hyb['not_relevant_pct']:.1%}")
    
    # Comparison
    if 'comparison' in metrics:
        print("\n" + "─"*70)
        print("COMPARISON (Hybrid vs Semantic)")
        print("─"*70)
        comp = metrics['comparison']
        
        prec_diff = comp['precision_improvement']
        rating_diff = comp['rating_improvement']
        
        print(f"\nPrecision@5 Improvement: {prec_diff:+.1%}")
        if prec_diff > 0:
            print(f"  → Hybrid is {prec_diff:.1%} better")
        elif prec_diff < 0:
            print(f"  → Semantic is {abs(prec_diff):.1%} better")
        else:
            print(f"  → No difference")
        
        print(f"\nMean Rating Improvement: {rating_diff:+.2f}")
        if rating_diff > 0:
            print(f"  → Hybrid produces better results")
        elif rating_diff < 0:
            print(f"  → Semantic produces better results")
        else:
            print(f"  → No difference")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# STEP 4: CREATE COMPARATIVE VISUALIZATIONS
# ============================================================================

def create_comparative_visualizations(df, output_dir='evaluation_results'):
    """Create comparison graphs for semantic vs hybrid."""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    sns.set_style("whitegrid")
    
    # ========================================================================
    # GRAPH 1: Side-by-Side Rating Distribution
    # ========================================================================
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (mode, ax) in enumerate([('semantic', ax1), ('hybrid', ax2)]):
        mode_df = df[df['search_mode'] == mode]
        rating_counts = mode_df['rating'].value_counts().sort_index()
        
        counts = [
            rating_counts.get(0, 0),
            rating_counts.get(1, 0),
            rating_counts.get(2, 0)
        ]
        
        percentages = [count / len(mode_df) * 100 for count in counts]
        
        bars = ax.bar(
            ['Not Relevant\n(0)', 'Somewhat\n(1)', 'Highly Relevant\n(2)'],
            counts,
            color=['#e74c3c', '#f39c12', '#27ae60'],
            edgecolor='black',
            linewidth=1.5
        )
        
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Number of Results', fontweight='bold')
        ax.set_title(f'{mode.upper()} Search\n(n={len(mode_df)})',
                     fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(counts) * 1.2)
    
    plt.suptitle('Rating Distribution: Semantic vs Hybrid Search',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_rating_distribution.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/comparison_rating_distribution.png")
    plt.close()
    
    # ========================================================================
    # GRAPH 2: Rating by Rank - Both Modes
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get ratings by rank for each mode
    semantic_by_rank = df[df['search_mode'] == 'semantic'].groupby('rank')['rating'].mean()
    hybrid_by_rank = df[df['search_mode'] == 'hybrid'].groupby('rank')['rating'].mean()
    
    x = range(1, 6)
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], semantic_by_rank, width,
                   label='Semantic', color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar([i + width/2 for i in x], hybrid_by_rank, width,
                   label='Hybrid', color='#9b59b6', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.axhline(y=1.0, color='#f39c12', linestyle='--', linewidth=2, alpha=0.5)
    ax.axhline(y=2.0, color='#27ae60', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Rank Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Rating', fontsize=12, fontweight='bold')
    ax.set_title('Result Quality by Rank: Semantic vs Hybrid',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 2.2)
    ax.set_xticks(x)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_rating_by_rank.png',
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/comparison_rating_by_rank.png")
    plt.close()
    
    # ========================================================================
    # GRAPH 3: Overall Comparison Bar Chart
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    semantic_metrics = {
        'Precision@5': df[df['search_mode'] == 'semantic']['rating'].apply(lambda x: 1 if x >= 1 else 0).mean(),
        'Mean Rating': df[df['search_mode'] == 'semantic']['rating'].mean() / 2.0,  # Normalize to 0-1
        'Highly Relevant %': (df[df['search_mode'] == 'semantic']['rating'] == 2).mean(),
    }
    
    hybrid_metrics = {
        'Precision@5': df[df['search_mode'] == 'hybrid']['rating'].apply(lambda x: 1 if x >= 1 else 0).mean(),
        'Mean Rating': df[df['search_mode'] == 'hybrid']['rating'].mean() / 2.0,
        'Highly Relevant %': (df[df['search_mode'] == 'hybrid']['rating'] == 2).mean(),
    }
    
    x = range(len(semantic_metrics))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], semantic_metrics.values(), width,
                   label='Semantic', color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar([i + width/2 for i in x], hybrid_metrics.values(), width,
                   label='Hybrid', color='#9b59b6', edgecolor='black', linewidth=1.5)
    
    # Add percentage labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Comparison',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(semantic_metrics.keys(), fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_overall_metrics.png',
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/comparison_overall_metrics.png")
    plt.close()
    
    print(f"\n✓ All comparative visualizations saved!\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution for comparative evaluation."""
    
    print("\n" + "="*70)
    print("COMPARATIVE EVALUATION - SEMANTIC vs HYBRID")
    print("="*70 + "\n")
    
    # Load database and model
    print("Loading ChromaDB and model...")
    collection, model = load_existing_database()
    print("✓ Database and model loaded\n")
    
    # Run evaluation
    print(f"You will evaluate {len(test_queries)} queries × 5 results × 2 modes")
    print(f"Total ratings needed: {len(test_queries) * 5 * 2}\n")
    
    input("Press Enter to begin comparative evaluation...")
    
    df = run_evaluation_comparison(collection, model, test_queries, query_keywords)
    
    # Calculate metrics
    print("\nCalculating comparative metrics...")
    metrics = calculate_metrics_by_mode(df)
    
    # Print results
    print_metrics_comparison(metrics)
    
    # Create visualizations
    print("Creating comparative visualizations...")
    create_comparative_visualizations(df)
    
    # Save results
    print("Saving results...")
    df.to_csv('evaluation_results/comparative_results.csv', index=False)
    
    with open('evaluation_results/comparative_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    
    print("\n" + "="*70)
    print("✅ COMPARATIVE EVALUATION COMPLETE!")
    print("="*70)
    print("\nResults saved to: evaluation_results/")
    print("  - comparison_rating_distribution.png")
    print("  - comparison_rating_by_rank.png")
    print("  - comparison_overall_metrics.png")
    print("  - comparative_results.csv")
    print("  - comparative_metrics.json")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()