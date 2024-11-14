from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import boto3
from pymongo import MongoClient
from bson import json_util
import json
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
import html

app = Flask(__name__)
load_dotenv()

# Initialize clients
client = MongoClient(os.getenv('MONGODB_CONNECTION_STRING'))
db = client.sample_mflix
collection = db.movies

runtime_client = boto3.client(
    'sagemaker-runtime',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

# Initialize LangChain and LlamaIndex components
llm = OpenAI(temperature=0.7, openai_api_key=os.getenv('OPENAI_API_KEY'))
output_parser = StrOutputParser()

query_prompt = PromptTemplate.from_template("Enhance this movie search query for better semantic search: {query}")
mood_prompt = PromptTemplate.from_template("Analyze the mood and emotional elements of this plot: {plot}")
theme_prompt = PromptTemplate.from_template("Identify the main themes and genres present in this plot: {plot}")

query_chain = query_prompt | llm | output_parser
mood_chain = mood_prompt | llm | output_parser
theme_chain = theme_prompt | llm | output_parser

def sanitize_string(text):
    """Sanitize string for safe JSON encoding"""
    if text is None:
        return ""
    # Escape HTML characters and remove problematic characters
    return html.escape(str(text)).encode('ascii', 'xmlcharrefreplace').decode()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query = html.escape(data['query'].strip())
        if not query:
            return jsonify({'error': 'Empty query'}), 400

        # Get embedding for search
        try:
            payload = {"inputs": query}
            response = runtime_client.invoke_endpoint(
                EndpointName=os.getenv('SAGEMAKER_ENDPOINT'),
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            embedding_result = json.loads(response['Body'].read().decode())
            if isinstance(embedding_result, list):
                embedding = embedding_result[0]
            else:
                embedding = embedding_result['embedding'] if 'embedding' in embedding_result else embedding_result
            
            embedding = embedding[0] if isinstance(embedding[0], list) else embedding
        except Exception as e:
            print(f"Embedding error: {str(e)}")
            return jsonify({'error': 'Error generating embedding'}), 500

        # Perform vector search
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "queryVector": embedding,
                        "path": "plot_embedding",
                        "numCandidates": 200,
                        "limit": 6,
                        "index": "plot_vector_index"
                    }
                },
                {
                    "$project": {
                        "title": 1,
                        "plot": 1,
                        "year": 1,
                        "genres": 1,
                        "imdb": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = list(collection.aggregate(pipeline))
            
            # Sanitize and enhance results
            enhanced_results = []
            for movie in results:
                # Sanitize strings
                movie['title'] = sanitize_string(movie.get('title', ''))
                movie['plot'] = sanitize_string(movie.get('plot', ''))
                if movie.get('genres'):
                    movie['genres'] = [sanitize_string(g) for g in movie['genres']]
                
                # Add analysis if plot exists
                if movie.get('plot'):
                    try:
                        movie['mood_analysis'] = sanitize_string(mood_chain.invoke({"plot": movie['plot']}))
                        movie['theme_analysis'] = sanitize_string(theme_chain.invoke({"plot": movie['plot']}))
                    except Exception as e:
                        print(f"Analysis error: {str(e)}")
                
                enhanced_results.append(movie)
            
            # Convert to JSON-safe format
            safe_results = json.loads(json_util.dumps(enhanced_results))
            
            return jsonify({
                'results': safe_results,
                'query': query
            })
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return jsonify({'error': 'Error performing search'}), 500
            
    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)