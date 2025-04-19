from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import os

app = FastAPI(title="BERT PII Detection API", 
              description="API for detecting Personally Identifiable Information in text using BERT model",
              version="1.0.0")

# model path
BERT_MODEL_PATH = "Trained_models/pii_detector_model_bert"

# load BERT model
try:
    # load BERT model and tokenizer
    bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    # ensure model on CPU
    device = torch.device("cpu")
    bert_model = bert_model.to(device)
    
    print(f"BERT model loaded successfully from {BERT_MODEL_PATH}!")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    raise RuntimeError(f"Model loading failed: {str(e)}")

# request and response model
class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    has_pii: bool
    confidence: float
    
# root route
@app.get("/")
async def root():
    return {"message": "Welcome to the BERT PII detection API! Use the /predict endpoint to detect personal information in text."}

# health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# predict endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_pii(request: TextRequest):
    text = request.text
    
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # prepare input
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # predict
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # get result
        has_pii = bool(probabilities[0, 1] > 0.5)  # second category (index 1) represents PII
        confidence = float(probabilities[0, 1])    # confidence of PII category
        
        return {
            "text": text,
            "has_pii": has_pii,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# batch predict endpoint
class BatchTextRequest(BaseModel):
    texts: list[str]

class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchTextRequest):
    texts = request.texts
    
    if not texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty")
    
    results = []
    
    try:
        # batch process all texts
        batch_inputs = bert_tokenizer(texts, padding=True, truncation=True, 
                                     max_length=128, return_tensors="pt")
        
        with torch.no_grad():
            batch_outputs = bert_model(**batch_inputs)
            batch_probabilities = torch.nn.functional.softmax(batch_outputs.logits, dim=-1)
        
        # process each prediction result
        for i, text in enumerate(texts):
            has_pii = bool(batch_probabilities[i, 1] > 0.5)
            confidence = float(batch_probabilities[i, 1])
            
            results.append({
                "text": text,
                "has_pii": has_pii,
                "confidence": confidence
            })
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during batch prediction: {str(e)}")

# model info endpoint
@app.get("/model_info")
async def model_info():
    return {
        "model": {
            "name": "BERT for PII Detection",
            "path": BERT_MODEL_PATH,
            "type": "BertForSequenceClassification",
            "base_model": "bert-base-uncased"
        }
    }

# data item processing endpoint
class DataItem(BaseModel):
    dataset_id: str
    id: str
    value: str

class DataItemResponse(BaseModel):
    dataset_id: str
    data_id: str
    prediction: bool
    confidence: float

@app.post("/predict_item", response_model=DataItemResponse)
async def predict_violation(item: DataItem):
    # prepare input
    inputs = bert_tokenizer(item.value, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # predict
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # get prediction result
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    
    return {
        "dataset_id": item.dataset_id,
        "data_id": item.id,
        "prediction": bool(prediction),
        "confidence": torch.softmax(logits, dim=-1)[0][prediction].item()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)