import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

# mock bert output
class MockOutput:
    def __init__(self, logits):
        self.logits = logits

# mock model and tokenizer before creating test client
@pytest.fixture
def app_client():
    # mock BertForSequenceClassification
    with patch('transformers.BertForSequenceClassification.from_pretrained') as mock_model:
        # set model behavior
        model_instance = MagicMock()
        # mock model output - return an object with logits property
        model_instance.return_value = MockOutput(torch.tensor([[0.1, 0.9]]))
        mock_model.return_value = model_instance
        
        # mock BertTokenizer
        with patch('transformers.BertTokenizer.from_pretrained') as mock_tokenizer:
            # set tokenizer behavior
            tokenizer_instance = MagicMock()
            tokenizer_instance.return_value = {"input_ids": torch.tensor([[101, 2054, 2003, 1037, 3231, 102]]), 
                                              "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]])}
            mock_tokenizer.return_value = tokenizer_instance
            
            # import app
            from App.app import app
            
            # return test client
            return TestClient(app)

def test_root_endpoint(app_client):
    """test root endpoint"""
    response = app_client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "welcome" in response.json()["message"].lower()

def test_health_endpoint(app_client):
    """test health check endpoint"""
    response = app_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] == True

def test_predict_endpoint(app_client):
    """test predict endpoint"""
    # mock model output
    with patch('torch.nn.functional.softmax', return_value=torch.tensor([[0.1, 0.9]])):
        response = app_client.post(
            "/predict",
            json={"text": "This is a test text with PII information"}
        )
    
    assert response.status_code == 200
    result = response.json()
    assert result["text"] == "This is a test text with PII information"
    assert result["has_pii"] == True
    assert abs(result["confidence"] - 0.9) < 1e-6


def test_predict_empty_text(app_client):
    """test empty text case"""
    response = app_client.post(
        "/predict",
        json={"text": ""}
    )
    
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_batch_endpoint(app_client):
    """test batch predict endpoint"""
    # mock model output
    with patch('torch.nn.functional.softmax', return_value=torch.tensor([[0.9, 0.1], [0.2, 0.8]])):
        response = app_client.post(
            "/predict_batch",
            json={"texts": ["Text without PII", "Text with PII information"]}
        )
    
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["has_pii"] == False
    assert results[1]["has_pii"] == True

def test_model_info_endpoint(app_client):
    """test model info endpoint"""
    response = app_client.get("/model_info")
    assert response.status_code == 200
    assert "model" in response.json()
    assert response.json()["model"]["name"] == "BERT for PII Detection"

def test_predict_item_endpoint(app_client):
    """test predict item endpoint"""
    # mock model output
    with patch('torch.argmax', return_value=torch.tensor([1])):
        with patch('torch.softmax', return_value=torch.tensor([[0.1, 0.9]])):
            response = app_client.post(
                "/predict_item",
                json={
                    "dataset_id": "test_dataset",
                    "id": "item_123",
                    "value": "Text with PII information"
                }
            )
    
    assert response.status_code == 200
    result = response.json()
    assert result["dataset_id"] == "test_dataset"
    assert result["data_id"] == "item_123"
    assert result["prediction"] == True
    assert abs(result["confidence"] - 0.9) < 1e-6