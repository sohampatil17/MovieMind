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

app = Flask(__name__)
load_dotenv()

# Initialize MongoDB client
client = MongoClient(os.getenv('MONGODB_CONNECTION_STRING'))
db = client.sample_mflix
collection = db.movies

# Initialize SageMaker client
runtime_client = boto3.client(
    'sagemaker-runtime',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

# Initialize OpenAI and prompts
llm = OpenAI(temperature=0.7, openai_api_key=os.getenv('OPENAI_API_KEY'))
output_parser = StrOutputParser()

# Create LangChain prompts and chains
query_prompt = PromptTemplate.from_template("Enhance this movie search query for better semantic search: {query}")
mood_prompt = PromptTemplate.from_template("Analyze the mood and emotional elements of this plot: {plot}")
theme_prompt = PromptTemplate.from_template("Identify the main themes and genres present in this plot: {plot}")

query_chain = query_prompt | llm | output_parser
mood_chain = mood_prompt | llm | output_parser
theme_chain = theme_prompt | llm | output_parser

# Initialize LlamaIndex
def initialize_llama_index():
    """Initialize LlamaIndex with movie data"""
    try:
        movies = list(collection.find(
            {"plot": {"$exists": True}},
            {"_id": 0, "title": 1, "plot": 1, "genres": 1}
        ).limit(100))
        
        documents = []
        for movie in movies:
            genres = ', '.join(movie.get('genres', [])) if movie.get('genres') else 'Unknown'
            doc_text = f"Title: {movie['title']}\nGenres: {genres}\nPlot: {movie['plot']}"
            documents.append(Document(text=doc_text))
        
        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)
        
        return index
    except Exception as e:
        print(f"Error initializing LlamaIndex: {str(e)}")
        return None

llama_index = initialize_llama_index()

def get_embedding(text):
    """Generate embedding using SageMaker endpoint"""
    try:
        payload = {"inputs": text}
        response = runtime_client.invoke_endpoint(
            EndpointName=os.getenv('SAGEMAKER_ENDPOINT'),
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        response_body = json.loads(response['Body'].read().decode())
        
        if isinstance(response_body, list):
            embedding = response_body[0]
        else:
            embedding = response_body['embedding'] if 'embedding' in response_body else response_body
            
        return embedding[0] if isinstance(embedding[0], list) else embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise

def vector_search(query, limit=6):
    """Perform vector search in MongoDB"""
    try:
        query_embedding = get_embedding(query)
        print(f"\nQuery embedding length: {len(query_embedding)}")
        
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": query_embedding,
                    "path": "plot_embedding",
                    "numCandidates": 200,
                    "limit": limit,
                    "index": "plot_vector_index"
                }
            },
            {
                "$project": {
                    "title": 1,
                    "plot": 1,
                    "year": 1,
                    "poster": 1,
                    "genres": 1,
                    "imdb": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        return json.loads(json_util.dumps(results))
    except Exception as e:
        print(f"Error in vector search: {str(e)}")
        raise

def enhance_query(query):
    """Use LangChain to enhance search query"""
    try:
        return query_chain.invoke({"query": query}).strip()
    except Exception as e:
        print(f"Query enhancement failed: {str(e)}")
        return query

def analyze_movie_mood(plot):
    """Use LangChain to analyze movie mood"""
    try:
        return mood_chain.invoke({"plot": plot}).strip()
    except Exception as e:
        print(f"Mood analysis failed: {str(e)}")
        return None

def analyze_movie_themes(plot):
    """Use LangChain to analyze movie themes"""
    try:
        return theme_chain.invoke({"plot": plot}).strip()
    except Exception as e:
        print(f"Theme analysis failed: {str(e)}")
        return None

def find_similar_scenes(plot):
    """Use LlamaIndex to find similar scenes"""
    try:
        if not llama_index:
            return []
        
        query_engine = llama_index.as_query_engine(
            similarity_top_k=3,
            response_mode="no_text"
        )
        response = query_engine.query(plot)
        
        similar_scenes = []
        for node in response.source_nodes:
            title = node.text.split("Title: ")[1].split("\n")[0]
            genres = node.text.split("Genres: ")[1].split("\n")[0]
            similar_scenes.append({
                "title": title,
                "genres": genres.split(', '),
                "score": float(node.score) if node.score else 0
            })
        
        return similar_scenes
    except Exception as e:
        print(f"Scene search failed: {str(e)}")
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    try:
        # 1. Use LangChain to enhance the query
        enhanced_query = enhance_query(query)
        print(f"Original query: {query}")
        print(f"Enhanced query: {enhanced_query}")
        
        # 2. Use Vector Search for main results
        vector_results = vector_search(enhanced_query)
        print(f"Found {len(vector_results)} vector search results")
        
        # 3. Enhance results with LangChain and LlamaIndex
        enriched_results = []
        for movie in vector_results:
            if movie.get('plot'):
                # Add LangChain analysis
                movie['mood_analysis'] = analyze_movie_mood(movie['plot'])
                movie['theme_analysis'] = analyze_movie_themes(movie['plot'])
                # Add LlamaIndex scene analysis
                movie['similar_scenes'] = find_similar_scenes(movie['plot'])
            enriched_results.append(movie)
        
        return jsonify({
            'results': enriched_results,
            'enhanced_query': enhanced_query if enhanced_query != query else None,
            'technologies_used': {
                'vector_search': 'MongoDB Atlas with SageMaker embeddings',
                'query_enhancement': 'LangChain',
                'mood_analysis': 'LangChain',
                'theme_analysis': 'LangChain',
                'scene_analysis': 'LlamaIndex'
            }
        })
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)