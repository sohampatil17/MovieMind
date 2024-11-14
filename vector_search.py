import os
from dotenv import load_dotenv
import json
import boto3
from pymongo import MongoClient
import numpy as np
from tqdm import tqdm

# Load environment variables
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

def check_vector_search_setup():
    """Check vector search setup"""
    print("\nChecking vector search setup...")
    
    # Check for documents with embeddings
    embedding_count = collection.count_documents({"plot_embedding": {"$exists": True}})
    print(f"Documents with embeddings: {embedding_count}")
    
    # Check a sample document with embedding
    sample_doc = collection.find_one({"plot_embedding": {"$exists": True}})
    if sample_doc and 'plot_embedding' in sample_doc:
        embedding_length = len(sample_doc['plot_embedding'])
        print(f"Sample embedding dimension: {embedding_length}")
    else:
        print("No documents found with embeddings")
    
    # List search indexes
    try:
        indexes = collection.list_indexes()
        print("\nAvailable indexes:")
        for idx in indexes:
            print(f"- {idx['name']}: {idx['key']}")
    except Exception as e:
        print(f"Error listing indexes: {str(e)}")

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
        
        # Extract embedding from response
        if isinstance(response_body, list):
            embedding = response_body[0]
        else:
            embedding = response_body['embedding'] if 'embedding' in response_body else response_body
            
        # Ensure embedding is a flat list
        if isinstance(embedding, list) and isinstance(embedding[0], list):
            embedding = embedding[0]
            
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise

def generate_embeddings(batch_size=100):
    """Generate and store embeddings for movies without them"""
    try:
        # Find movies without embeddings but with plots
        movies_to_process = list(collection.find(
            {"plot": {"$exists": True}, "plot_embedding": {"$exists": False}},
            {"_id": 1, "plot": 1, "title": 1}
        ).limit(batch_size))
        
        if not movies_to_process:
            print("All movies already have embeddings!")
            return
        
        print(f"Generating embeddings for {len(movies_to_process)} movies...")
        
        # Generate and store embeddings
        for movie in tqdm(movies_to_process):
            try:
                if movie.get('plot'):
                    embedding = get_embedding(movie['plot'])
                    result = collection.update_one(
                        {"_id": movie['_id']},
                        {"$set": {"plot_embedding": embedding}}
                    )
                    if not result.modified_count:
                        print(f"Warning: Failed to update movie {movie.get('title', movie['_id'])}")
            except Exception as e:
                print(f"Error processing movie {movie.get('title', movie['_id'])}: {str(e)}")
                continue
                
        print("Embedding generation complete!")
        
    except Exception as e:
        print(f"Error in generate_embeddings: {str(e)}")
        raise

def vector_search(query, limit=5):
    """Perform vector search in MongoDB"""
    try:
        query_embedding = get_embedding(query)
        print(f"\nQuery embedding length: {len(query_embedding)}")
        
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": query_embedding,
                    "path": "plot_embedding",
                    "numCandidates": 200,  # Increased from 100
                    "limit": limit,
                    "index": "plot_vector_index"  # Make sure this matches your index name
                }
            },
            {
                "$project": {
                    "title": 1,
                    "plot": 1,
                    "year": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        print("\nExecuting search...")
        results = list(collection.aggregate(pipeline))
        print(f"Found {len(results)} results")
        return results
    except Exception as e:
        print(f"Error performing vector search: {str(e)}")
        raise

def main():
    try:
        # Check setup first
        check_vector_search_setup()
        
        # Generate embeddings if needed
        generate_embeddings()
        
        # Check setup again after generating embeddings
        check_vector_search_setup()
        
        # Perform search
        query = "A movie about artificial intelligence taking over the world"
        print("\nProcessing query:", query)
        results = vector_search(query)
        
        if not results:
            print("\nNo results found. This might mean either:")
            print("1. No similar movies were found")
            print("2. The vector search index isn't properly configured")
            print("3. No movies have embeddings stored yet")
        else:
            print("\nResults:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} ({result.get('year', 'N/A')})")
                print(f"Score: {result['score']:.4f}")
                print(f"Plot: {result.get('plot', 'No plot available')}")
                
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()