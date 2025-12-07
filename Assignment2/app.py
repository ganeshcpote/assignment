"""
Flask Web Application for the Personal Google Search Engine
"""

from flask import Flask, render_template, request, jsonify
from documents import DOCUMENTS
from search_engine import SearchEngine

app = Flask(__name__)

# Initialize the search engine
search_engine = SearchEngine(DOCUMENTS)


@app.route('/')
def index():
    """Render the main search page."""
    stats = search_engine.get_stats()
    return render_template('index.html', stats=stats)


@app.route('/search', methods=['POST'])
def search():
    """Handle search queries and return results."""
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    # Perform search
    results = search_engine.search(query, top_k=5)
    
    return jsonify({
        'query': query,
        'results': results,
        'total_results': len(results)
    })


@app.route('/stats', methods=['GET'])
def stats():
    """Get search engine statistics."""
    return jsonify(search_engine.get_stats())


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

