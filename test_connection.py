import os
from dotenv import load_dotenv
from pymongo import MongoClient
import boto3

def test_connections():
    # Load environment variables
    load_dotenv()
    
    # Test MongoDB connection
    print("Testing MongoDB connection...")
    try:
        client = MongoClient(os.getenv('MONGODB_CONNECTION_STRING'))
        db = client.sample_mflix
        count = db.movies.count_documents({})
        print(f"✓ MongoDB connected successfully! Found {count} documents in movies collection")
    except Exception as e:
        print(f"✗ MongoDB connection failed: {str(e)}")
    
    # Test AWS/SageMaker connection
    print("\nTesting AWS/SageMaker connection...")
    try:
        # Create both sagemaker and sagemaker-runtime clients
        sagemaker = boto3.client('sagemaker',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        
        runtime_client = boto3.client('sagemaker-runtime',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        
        # List endpoints using sagemaker client
        endpoints = sagemaker.list_endpoints()
        print("✓ AWS/SageMaker connection successful!")
        print("\nAvailable endpoints:")
        for endpoint in endpoints['Endpoints']:
            print(f"- {endpoint['EndpointName']}")
            
    except Exception as e:
        print(f"✗ AWS/SageMaker connection failed: {str(e)}")

if __name__ == "__main__":
    test_connections()