# Human_Native

# PII Detection API

## Overview

This API provides a service for detecting Personally Identifiable Information (PII) in text using a fine-tuned BERT model. It can identify sensitive information such as names, email addresses, phone numbers, and other personal data that might need to be redacted or handled with care.

## Features

- Fast and accurate PII detection
- REST API interface
- Containerized deployment with Docker
- Built with FastAPI for high performance
- Powered by a fine-tuned BERT model

## Getting Started

### Prerequisites

- Docker
- Docker Compose (optional)

### Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Tofu0142/Human_Native.git
   cd pii-detection-api
   ```

2. Build the Docker image:
   ```bash
   docker build -t pii-detector-api .
   ```

3. Run the container:
   ```bash
   docker run -p 8000:8000 pii-detector-api
   ```

The API will be available at `http://localhost:8000`.

## API Usage

### Detect PII in Text

**Endpoint:** `/predict`

**Method:** POST

**Request Body:**
```json
{
  "text": "My email is john.doe@example.com and my phone is 555-123-4567"
}
```

**Response:**
```json
{
  
  "redacted_text": "My name is [NAME] and my email is [EMAIL]",
  "has_pii": true,
  "confidence": 0.9999833106994629
}
```

### Example Usage with curl

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "My email is john.doe@example.com and my phone is 555-123-4567"
}'
```
### Interactive API Documentation
FastAPI automatically generates interactive API documentation:
1. Open a web browser
2. Navigate to http://localhost:8000/docs
3. You'll see the Swagger UI where you can explore and test all endpoints



## Development

### Local Development Setup

1. Install Poetry:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Run the application:
   ```bash
   poetry run uvicorn App.app:app --reload
   ```

### Running Tests

```bash
poetry run pytest
```

## Data Generation and Training
The models were trained on a synthetic dataset generated to include various types of PII:
1. Data Generation: We used a custom data generation pipeline to create realistic text samples with and without PII
2. Training Process:
    - Random Forest: Trained on feature-engineered text data + TF-IDF + Rule-based detection
    - BERT: Fine-tuned on raw text with binary labels
3. Model Selection:
    After evaluation, we selected the BERT model for its higher accuracy and performance in identifying common PII types.
### Model Comparison
Our evaluation shows the performance comparison between the Random Forest and BERT models:
| Metric | Random Forest | BERT |
|--------|--------------|------|
| Accuracy | 0.943000 | 1.000000 |
| Precision | 0.873503 | 1.000000 |
| Recall | 0.951876 | 1.000000 |
| F1 Score | 0.911007 | 1.000000 |
| AUC | 0.985578 | 1.000000 |
| Inference Time | 32.590107 | 29.393385 |

The BERT model provides higher accuracy and with lower inference time, so we selected the BERT model for the API.

## Assumptions and Design Decisions
1. Dual Model Approach: We implemented both a deep learning model (BERT) and a traditional ML model (Random Forest) to provide options for different use cases.
2. Binary Classification: The models perform binary classification (PII/No PII) rather than multi-class classification of specific PII types.
3. Confidence Score: The API returns a confidence score to allow users to set their own thresholds for PII detection.
4. Stateless API: The API is designed to be stateless, making it easy to scale horizontally.
5. Docker Deployment: The solution is containerized for easy deployment in various environments.
6. No Persistent Storage: The API doesn't store any of the processed text, ensuring privacy.
7. Performance Optimization: The models are loaded once at startup to minimize inference time.
8. Error Handling: The API includes robust error handling for empty texts and other edge cases. 

##Future Work and Considerations
### Model Architecture and Performance
1. Hybrid Inference Strategy: Implement a cascading approach where the faster Random Forest model performs initial screening, and the more accurate BERT model only processes uncertain cases.
2. Model Optimization: Explore model quantization, distillation, or smaller pre-trained models like DistilBERT to reduce model size and improve inference speed.
### Data and Training Strategies
1. Synthetic Data Limitations: While our models perform excellently on synthetic data, real-world text may present additional challenges. Consider generating more diverse synthetic data or incorporating real-world examples.
2. Class Imbalance: Evaluate and address potential class imbalance issues using techniques like oversampling, undersampling, or weighted loss functions.
### Monitoring and Continuous Improvement
1. Model Performance Monitoring: Implement systems to track key metrics over time and design feedback mechanisms for users to report false positives/negatives.
2. A/B Testing Framework: Design a framework to safely introduce model improvements by comparing multiple model versions simultaneously.
### Extended Application Scenarios
1. Multilingual Support: Explore multilingual BERT models or language-specific models to support PII detection across different languages.

## License

[MIT License](LICENSE)

