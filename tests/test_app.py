import pytest
from unittest.mock import AsyncMock, patch
from app import chat
from datetime import datetime

@pytest.mark.asyncio
async def test_chat_empty_collection(mocker):
    mock_collection = AsyncMock()
    mock_collection.count_documents.return_value = 0
    mock_collection.insert_many = AsyncMock()
    
    mock_chat_request = AsyncMock()
    mock_chat_request.query = "Show sales trends"

    with patch('app.collection', mock_collection):
        response = await chat(mock_chat_request)

    assert mock_collection.insert_many.called
    assert response['response'] is not None

@pytest.mark.asyncio
async def test_chat_with_visualization_query(mocker):
    mock_collection = AsyncMock()
    mock_collection.count_documents.return_value = 1
    mock_collection.aggregate.return_value = [
        {"document": "CUST1 TRX1 PROD1 100.0 City1"}
    ]
    
    mock_chat_request = AsyncMock()
    mock_chat_request.query = "Plot the amount by city"

    with patch('app.collection', mock_collection):
        with patch('app.visualize', return_value="Visualization Response") as mock_visualize:
            response = await chat(mock_chat_request)

    assert mock_visualize.called
    assert response['response'] == "Visualization Response"

@pytest.mark.asyncio
async def test_chat_with_regular_query(mocker):
    mock_collection = AsyncMock()
    mock_collection.count_documents.return_value = 1
    mock_collection.aggregate.return_value = [
        {"document": "CUST1 TRX1 PROD1 100.0 City1"}
    ]
    
    mock_chat_request = AsyncMock()
    mock_chat_request.query = "Tell me about customer CUST1"

    with patch('app.collection', mock_collection):
        with patch('app.get_embeddings', return_value=[0.1, 0.2, 0.3]) as mock_get_embeddings:
            with patch('app.cosine_similarity', return_value=[[1.0]]) as mock_cosine_similarity:
                with patch('app.generate_with_ollama', return_value="Response with context") as mock_generate:
                    response = await chat(mock_chat_request)

    assert mock_get_embeddings.called
    assert mock_cosine_similarity.called
    assert mock_generate.called
    assert response['response'] == "Response with context"

@pytest.mark.asyncio
async def test_chat_no_documents(mocker):
    mock_collection = AsyncMock()
    mock_collection.count_documents.return_value = 1
    mock_collection.aggregate.return_value = []

    mock_chat_request = AsyncMock()
    mock_chat_request.query = "What is the revenue?"

    with patch('app.collection', mock_collection):
        with patch('app.generate_with_ollama', return_value="Response without context") as mock_generate:
            response = await chat(mock_chat_request)

    assert mock_generate.called
    assert response['response'] == "Response without context"
