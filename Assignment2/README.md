# Personal Google Search Engine

A simple yet powerful search engine that demonstrates the fundamentals of information retrieval using TF-IDF vectorization and cosine similarity. This project is designed to help understand how search engines work and serves as a foundation for RAG (Retrieval-Augmented Generation) systems.

## Features

- **Document Chunking**: Automatically breaks documents into chunks of 3 consecutive sentences using NLTK
- **TF-IDF Vectorization**: Converts text into numerical vectors that capture term importance
- **Cosine Similarity**: Ranks documents by semantic relevance, not just keyword matching
- **Web Interface**: Beautiful, modern web app for interactive searching
- **Top 5 Results**: Displays the most relevant document chunks ranked by similarity score

## How It Works

1. **Document Processing**: Documents are tokenized into sentences and grouped into chunks (3 sentences each)
2. **Vectorization**: Both queries and document chunks are converted to TF-IDF vectors
3. **Similarity Calculation**: Cosine similarity measures how similar the query vector is to each chunk vector
4. **Ranking**: Chunks are sorted by similarity score (highest to lowest)
5. **Results**: Top 5 most relevant chunks are displayed with their scores

## Installation

1. Install Python 3.13 (or compatible version)

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Enter a search query in the search box (e.g., "AI and machine learning")
2. Click "Search" or press Enter
3. View the top 5 most relevant document chunks ranked by similarity score
4. Each result shows:
   - Document title
   - Similarity score (as a percentage)
   - The actual chunk content
   - Metadata (document ID and chunk index)

## Example Queries

Try these queries to see how the search engine works:

- "AI and machine learning" - Should match documents about AI tools
- "cloud infrastructure AWS Azure" - Should match the cloud infrastructure document
- "finance revenue profit" - Should match the finance report
- "SEO marketing campaign" - Should match the marketing document
- "analytics data insights" - Should match the AI analytics tool document

## Project Structure

```
.
├── app.py                 # Flask web application
├── search_engine.py       # Core search engine implementation
├── documents.py           # Sample documents database
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Web interface
└── README.md             # This file
```

## Technical Details

### TF-IDF (Term Frequency-Inverse Document Frequency)
- **Term Frequency (TF)**: How often a term appears in a document
- **Inverse Document Frequency (IDF)**: How rare/common a term is across all documents
- **TF-IDF**: Combines both to give higher weight to terms that are frequent in a document but rare across the corpus

### Cosine Similarity
- Measures the cosine of the angle between two vectors
- Range: 0 (completely different) to 1 (identical)
- Works well for text similarity because it's normalized by vector magnitude

### Why This Matters (The CTO Takeaway)

1. **It's Not Just Keyword Matching**: The system understands semantic relationships, not just exact word matches
2. **Foundation of RAG**: This is the "R" (Retrieval) in RAG - finding relevant information before generating answers
3. **Scalable Approach**: This classic method is fast, efficient, and still widely used in production systems

## Use Cases

- **Internal Wiki Search**: Fast search for company documentation
- **Knowledge Base**: Search through technical documentation
- **Content Discovery**: Find relevant articles or documents
- **RAG Systems**: Use as the retrieval component in a larger AI system

## Future Enhancements

- Add more documents to the database
- Implement document upload functionality
- Add filters (by document type, date, etc.)
- Support for multiple languages
- Advanced ranking algorithms (BM25, etc.)
- Query expansion and synonym handling

## License

This project is for educational purposes.

