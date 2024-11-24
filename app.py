from functools import lru_cache
from fastapi import FastAPI, Request, HTTPException
from pymongo import MongoClient
from pydantic import BaseModel
from typing import Union, List, Optional
import requests
import redis
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import os
from datetime import datetime, date
from dotenv import load_dotenv
import logging
from fastapi import Query
import random
from datetime import timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('barbarik_api.log'),
        logging.StreamHandler()
    ]
)

# Create plots directory
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

logger = logging.getLogger('barbarik_api')
# Load environment variables
load_dotenv()

# Pydantic models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

class VisualizationResponse(BaseModel):
    image: str
    file_path: str  # Added to return the saved file path

class TransactionData(BaseModel):
    customer_id: str
    transaction_id: str
    product_id: str
    timestamp: datetime
    amount: float
    city: str
    sales_volume: int

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST001",
                "transaction_id": "TRX001",
                "product_id": "PROD001",
                "timestamp": "2024-01-22T10:30:00",
                "amount": 299.99,
                "city": "New York",
                "sales_volume": 5
            }
        }

# Initialize FastAPI
app = FastAPI(
    title="BarbariK_Assignment",
    description="API Documentation for BarbariK Assignment",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Chat",
            "description": "Chat operations and query processing"
        },
        {
            "name": "Visualization",
            "description": "Data visualization endpoints"
        },
        {
            "name": "Data",
            "description": "Transaction data operations"
        },
        {
            "name": "Testing",
            "description": "Performance testing endpoints"
        }
    ]
)

# MongoDB connection
MONGO_URI = os.environ.get('MONGO_URI')
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client['chatbot_db']
collection = db['RAG_ASSIGNMENT_DATA']

# Create indexes
collection.create_index([
    ("customer_id", 1),
    ("timestamp", -1),
    ("city", 1)
])

# Redis configuration
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'decode_responses': True,
    'socket_timeout': 5,
    'retry_on_timeout': True
}

# Initialize Redis with connection pool
redis_pool = redis.ConnectionPool(**REDIS_CONFIG)
cache = redis.Redis(connection_pool=redis_pool)

# Initialize SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api/generate"

def get_cache_key(query_or_params):
    """Generate consistent cache keys"""
    if isinstance(query_or_params, str):
        return f"query_{hash(query_or_params)}"
    return f"viz_{hash(str(query_or_params))}"

def set_cache(key, value, expiry=3600):
    """Reliable cache setting with error handling"""
    try:
        if isinstance(value, dict):
            value = json.dumps(value)
        return cache.set(key, value, ex=expiry)
    except redis.RedisError as e:
        logger.error(f"Redis cache set failed: {e}")
        return False

def get_cache(key):
    """Reliable cache retrieval with error handling"""
    try:
        value = cache.get(key)
        if value and key.startswith('viz_'):
            return json.loads(value)
        return value
    except redis.RedisError as e:
        logger.error(f"Redis cache get failed: {e}")
        return None

@lru_cache(maxsize=1000)
def get_embeddings(text):
    return model.encode([text])[0]

@lru_cache(maxsize=100)
def generate_with_ollama(prompt, context=None):
    system_message = "You are a helpful assistant that answers questions based on the provided context."
    if context:
        system_message += f"\nContext: {context}"

    response = requests.post(OLLAMA_API, json={
        "model": "llama3",
        "prompt": f"{system_message}\n\nUser: {prompt}\nAssistant:",
        "stream": False
    })

    response_data = response.json()
    return response_data.get('response', 'I am processing your request.')

def generate_visualization(category: str, metric: str):
    """Generate and save visualization, returns both base64 image and file path"""
    pipeline = []
    if category == "time":
        pipeline = [
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "value": {"$sum": f"${metric}"}
            }},
            {"$sort": {"_id": 1}}
        ]
    else:
        pipeline = [
            {"$group": {
                "_id": f"${category}",
                "value": {"$sum": f"${metric}"}
            }},
            {"$match": {"_id": {"$ne": None}}}
        ]

    data = list(collection.aggregate(pipeline))
    
    plt.figure(figsize=(12, 6))
    labels = [str(item['_id']) if item['_id'] is not None else 'Unknown' for item in data]
    values = [float(item['value']) for item in data]

    if category == "time":
        plt.plot(labels, values, marker='o', linestyle='-')
    else:
        plt.bar(labels, values)

    plt.title(f"{metric.replace('_', ' ').title()} by {category.title()}")
    plt.xlabel(category.title())
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{category}_{metric}_{timestamp}.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    
    # Save the plot to file
    plt.savefig(filepath, format="png", dpi=300, bbox_inches='tight')
    
    # Generate base64 for API response
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()
    buffer.close()

    return f"data:image/png;base64,{image_base64}", filepath

@app.post("/init-database", tags=["Data"])
async def initialize_database():
    # Clear existing data
    collection.delete_many({})
    
    # Generate realistic sample data
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia"]
    products = ["Laptop", "Smartphone", "Tablet", "Desktop", "Printer", "Monitor"]
    
    sample_data = []
    for i in range(1000):
        sample_data.append({
            "customer_id": f"CUST{i:04d}",
            "transaction_id": f"TRX{i:06d}",
            "product_id": f"PROD_{random.choice(products)}",
            "timestamp": datetime.now() - timedelta(days=random.randint(0, 365)),
            "amount": round(random.uniform(100, 2000), 2),
            "city": random.choice(cities),
            "sales_volume": random.randint(1, 10)
        })
    
    # Insert the data
    collection.insert_many(sample_data)
    
    return {
        "status": "success",
        "records_inserted": len(sample_data),
        "message": "Database initialized with sample data"
    }

@app.post("/chat", response_model=Union[ChatResponse, VisualizationResponse], tags=["Chat"])
async def chat(chat_request: ChatRequest):
    user_query = chat_request.query.lower()
    cache_key = get_cache_key(user_query)

    # Check cache first
    cached_response = get_cache(cache_key)
    if cached_response:
        logger.info("Cache hit for query")
        return {"response": cached_response}

    viz_keywords = {
        'show', 'plot', 'graph', 'visualize', 'display', 'chart', 'trend', 'compare'
    }

    if any(keyword in user_query for keyword in viz_keywords):
        category_mapping = {
            'time': ['time', 'date', 'period', 'temporal', 'when'],
            'city': ['city', 'cities', 'location', 'place', 'where'],
            'customer_id': ['customer', 'customers', 'buyer', 'client'],
            'product_id': ['product', 'products', 'item', 'goods']
        }

        detected_category = None
        for system_category, keywords in category_mapping.items():
            if any(keyword in user_query for keyword in keywords):
                detected_category = system_category
                break
    
        if not detected_category:
            detected_category = 'time'

        metric = 'amount' if any(word in user_query for word in ['amount', 'revenue', 'money', 'price']) else 'sales_volume'

        logger.info(f"Processing visualization - Category: {detected_category}, Metric: {metric}")
        return await visualize(category=detected_category, metric=metric)

    # Handle regular chat queries
    query_embedding = get_embeddings(user_query)

    pipeline = [
        {
            "$project": {
                "document": {
                    "$concat": [
                        {"$toString": "$customer_id"}, " ",
                        {"$toString": "$transaction_id"}, " ",
                        {"$toString": "$product_id"}, " ",
                        {"$toString": "$amount"}, " ",
                        "$city"
                    ]
                }
            }
        }
    ]

    documents = list(collection.aggregate(pipeline))

    if documents:
        doc_texts = [doc['document'] for doc in documents]
        doc_embeddings = [get_embeddings(text) for text in doc_texts]
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        relevant_context = "\n".join([doc_texts[i] for i in top_indices])
        final_response = generate_with_ollama(user_query, relevant_context)
    else:
        final_response = generate_with_ollama(user_query)

    set_cache(cache_key, final_response)
    return {"response": final_response}

@app.get("/visualization", response_model=VisualizationResponse, tags=["Visualization"])
async def visualize(
    category: str = Query(..., description="Category to visualize: city, product_id, customer_id, or time"),
    metric: str = Query("sales_volume", description="Metric to analyze: sales_volume or amount")
):
    category = category.lower()
    if category not in {"city", "product_id", "customer_id", "time"}:
        raise HTTPException(status_code=400, detail="Invalid category")
    if metric not in {"sales_volume", "amount"}:
        raise HTTPException(status_code=400, detail="Invalid metric")

    image_data, file_path = generate_visualization(category, metric)
    return {"image": image_data, "file_path": file_path}

@app.get("/transactions", response_model=List[TransactionData], tags=["Data"])
async def get_transactions(
    limit: int = Query(100, description="Number of records to return"),
    city: Optional[str] = Query(None, description="City name to filter"),
    start_date: Optional[date] = Query(None, description="Start date in YYYY-MM-DD format")
):
    logger.info(f"Querying transactions - city: {city}, start_date: {start_date}")
    
    query = {}
    if city:
        query["city"] = city
    if start_date:
        date_obj = datetime.combine(start_date, datetime.min.time())
        query["timestamp"] = {"$gte": date_obj}

    result = collection.find(query, {'_id': 0}).limit(limit)
    return list(result)

@app.post("/stress-test", tags=["Testing"])
async def stress_test(num_records: int = 1000):
    start_time = datetime.now()
    
    test_data = [
        TransactionData(
            customer_id=f"CUST{i}",
            transaction_id=f"TRX{i}",
            product_id=f"PROD{i%100}",
            timestamp=datetime.now(),
            amount=float(i * 10),
            city=f"City{i%10}",
            sales_volume=i%50
        ).dict() for i in range(num_records)
    ]
    
    collection.insert_many(test_data)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    return {
        "records_inserted": num_records,
        "execution_time": execution_time,
        "records_per_second": num_records/execution_time
    }

@app.on_event("startup")
def startup_event():
    logger.info("Starting application")
    logger.info("Creating indexes...")
    collection.create_index([("city", 1), ("timestamp", 1)])
    logger.info("Connecting to MongoDB Atlas...")
    logger.info("Connecting to Ollama service...")
    logger.info("Connecting to Redis cache...")

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to BarbariK Assignment API",
        "version": "1.0.0",
        "endpoints": {
            "documentation": "/docs",
            "chat": "/chat",
            "transactions": "/transactions",
            "visualization": "/visualization",
            "stress_test": "/stress-test"
        },
        "status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)