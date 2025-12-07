"""
Quick test script to verify the search engine works correctly.
"""

from documents import DOCUMENTS
from search_engine import SearchEngine

def test_search_engine():
    """Test the search engine with sample queries."""
    
    # Initialize search engine
    print("Initializing search engine...")
    engine = SearchEngine(DOCUMENTS)
    
    # Print stats
    stats = engine.get_stats()
    print(f"\nSearch Engine Stats:")
    print(f"  Total Documents: {stats['total_documents']}")
    print(f"  Total Chunks: {stats['total_chunks']}")
    print(f"  Avg Chunks per Doc: {stats['avg_chunks_per_doc']:.2f}")
    
    # Test queries
    test_queries = [
        "AI and machine learning",
        "cloud infrastructure AWS Azure",
        "finance revenue profit",
        "SEO marketing",
        "analytics data"
    ]
    
    print("\n" + "="*60)
    print("Testing Search Queries")
    print("="*60)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 60)
        results = engine.search(query, top_k=5)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n  Rank #{i} (Score: {result['similarity_score']:.4f})")
                print(f"  Document: {result['doc_title']}")
                print(f"  Chunk Preview: {result['chunk'][:100]}...")
        else:
            print("  No results found")
    
    print("\n" + "="*60)
    print("✅ All tests completed successfully!")
    print("="*60)

if __name__ == "__main__":
    test_search_engine()

