"""
Search Engine Implementation
Uses TF-IDF vectorization and cosine similarity to rank document chunks.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Try to use NLTK, fallback to regex if not available
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    
    # Try to download NLTK data, but don't fail if it doesn't work
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except (LookupError, AttributeError):
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            try:
                nltk.data.find('tokenizers/punkt')
            except (LookupError, AttributeError):
                try:
                    nltk.download('punkt', quiet=True)
                except:
                    pass  # Will use regex fallback
    
    USE_NLTK = True
except ImportError:
    USE_NLTK = False


def sent_tokenize(text):
    """
    Tokenize text into sentences.
    Uses NLTK if available, otherwise falls back to regex-based splitting.
    """
    if USE_NLTK:
        try:
            from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
            return nltk_sent_tokenize(text)
        except:
            pass
    
    # Regex-based sentence tokenization fallback
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]


class SearchEngine:
    """
    A simple search engine that:
    1. Chunks documents into groups of 3 consecutive sentences
    2. Vectorizes queries and chunks using TF-IDF
    3. Calculates cosine similarity
    4. Ranks and returns top results
    """
    
    def __init__(self, documents):
        """
        Initialize the search engine with documents.
        
        Args:
            documents: List of dicts with 'id', 'title', and 'content' keys
        """
        self.documents = documents
        self.chunks = []
        self.chunk_metadata = []  # Store which document each chunk belongs to
        self.vectorizer = None
        self.chunk_vectors = None
        
        # Process documents into chunks
        self._create_chunks()
        # Vectorize chunks
        self._vectorize_chunks()
    
    def _create_chunks(self):
        """
        Break documents into chunks of 3 consecutive sentences using NLTK.
        """
        self.chunks = []
        self.chunk_metadata = []
        
        for doc in self.documents:
            # Tokenize document into sentences
            sentences = sent_tokenize(doc['content'].strip())
            
            # Group every 3 consecutive sentences
            for i in range(0, len(sentences), 3):
                chunk = ' '.join(sentences[i:i+3])
                if chunk.strip():  # Only add non-empty chunks
                    self.chunks.append(chunk)
                    self.chunk_metadata.append({
                        'doc_id': doc['id'],
                        'doc_title': doc['title'],
                        'chunk_index': len(self.chunks) - 1,
                        'sentence_range': (i, min(i+3, len(sentences)))
                    })
    
    def _vectorize_chunks(self):
        """
        Create TF-IDF vectors for all chunks.
        """
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=5000
        )
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
    
    def search(self, query, top_k=5):
        """
        Search for the most relevant chunks to the query.
        
        Args:
            query: User's search query string
            top_k: Number of top results to return (default: 5)
        
        Returns:
            List of dicts with chunk content, similarity score, and metadata
        """
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between query and all chunks
        similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        
        # Get top_k indices (sorted by similarity, descending)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            similarity_score = float(similarities[idx])
            if similarity_score > 0:  # Only include chunks with non-zero similarity
                metadata = self.chunk_metadata[idx].copy()
                results.append({
                    'chunk': self.chunks[idx],
                    'similarity_score': similarity_score,
                    'doc_id': metadata['doc_id'],
                    'doc_title': metadata['doc_title'],
                    'chunk_index': metadata['chunk_index']
                })
        
        return results
    
    def get_stats(self):
        """
        Get statistics about the search engine.
        
        Returns:
            Dict with statistics
        """
        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'avg_chunks_per_doc': len(self.chunks) / len(self.documents) if self.documents else 0
        }

