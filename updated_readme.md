# SvaraAI Reply Classification API

A production-ready FastAPI sentiment classification service that analyzes text replies and returns sentiment predictions with confidence scores. Built for Windows deployment with Logistic Regression model support.

## Overview

This project provides a complete sentiment analysis solution with:
- **Training Pipeline**: Jupyter notebook for model development 
- **FastAPI Service**: Production-ready REST API with automatic model loading
- **Logistic Regression Model**: Fast, efficient sentiment classification
- **Windows Compatible**: Tested and optimized for Windows environments

## Features

- Single & batch text sentiment predictions
- Interactive API documentation with Swagger UI
- Comprehensive health monitoring and model information endpoints
- Professional HTML homepage for easy client onboarding
- Robust error handling and input validation
- Automatic model loading with fallback support

## Quick Start

### Prerequisites

```bash
pip install fastapi uvicorn scikit-learn joblib torch transformers numpy pandas pydantic
```

### Installation & Setup

1. **Ensure your folder structure matches:**
```
project/
├── app.py
├── pipeline.ipynb
├── reply_classification_dataset.csv
└── best_model/
    ├── label_encoder.pkl
    ├── lr_model.pkl
    └── tfidf_vectorizer.pkl
```

2. **Start the API server:**
```bash
python app.py
```

3. **Access your API:**
- **Homepage**: http://localhost:8000/ (user-friendly interface)
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Usage

### Single Text Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is amazing!"}'
```

**Response:**
```json
{
  "text": "This product is amazing!",
  "label": "positive",
  "confidence": 0.8542,
  "model_used": "Logistic Regression"
}
```

### Batch Text Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/batch_predict" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great service!", "Terrible experience", "It was okay"]}'
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "Great service!",
      "label": "positive",
      "confidence": 0.9234,
      "model_used": "Logistic Regression",
      "status": "success"
    },
    {
      "text": "Terrible experience",
      "label": "negative", 
      "confidence": 0.8901,
      "model_used": "Logistic Regression",
      "status": "success"
    },
    {
      "text": "It was okay",
      "label": "neutral",
      "confidence": 0.7543,
      "model_used": "Logistic Regression",
      "status": "success"
    }
  ],
  "total_count": 3,
  "successful_count": 3,
  "failed_count": 0,
  "model_used": "Logistic Regression"
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_type": "Logistic Regression",
  "classes": ["negative", "neutral", "positive"],
  "device": "cpu"
}
```

### Model Information

```bash
curl http://localhost:8000/model_info
```

**Response:**
```json
{
  "model_type": "Logistic Regression",
  "classes": ["negative", "neutral", "positive"],
  "num_classes": 3,
  "device": "cpu",
  "model_files": {
    "directory": "./best_model",
    "files": ["label_encoder.pkl", "lr_model.pkl", "tfidf_vectorizer.pkl"]
  }
}
```

## Python Client Example

```python
import requests
import json

BASE_URL = "http://localhost:8000"

def predict_sentiment(text):
    """Predict sentiment for a single text"""
    response = requests.post(f"{BASE_URL}/predict", json={"text": text})
    return response.json()

def predict_batch(texts):
    """Predict sentiment for multiple texts"""
    response = requests.post(f"{BASE_URL}/batch_predict", json={"texts": texts})
    return response.json()

# Example usage
result = predict_sentiment("This product is amazing!")
print(f"Prediction: {result['label']} (confidence: {result['confidence']:.3f})")

# Batch prediction
batch_results = predict_batch([
    "Great quality!",
    "Not satisfied with the service",
    "Average experience"
])

for pred in batch_results['predictions']:
    print(f"'{pred['text']}' → {pred['label']} ({pred['confidence']:.3f})")
```

## Project Structure

```
svaraai-classification-api/
│
├── app.py                           # FastAPI application (main service)
├── pipeline.ipynb                   # Jupyter notebook for model training
├── reply_classification_dataset.csv # Training dataset
│
└── best_model/                      # Trained model files
    ├── label_encoder.pkl            # Label encoder (required)
    ├── lr_model.pkl                # Logistic Regression model
    └── tfidf_vectorizer.pkl        # TF-IDF vectorizer
```

## Model Information

### Current Implementation
- **Primary Model**: Logistic Regression with TF-IDF vectorization
- **Classes**: negative, neutral, positive
- **Features**: Text preprocessing with URL/email removal and normalization
- **Performance**: Fast inference suitable for production use

### Model Loading Priority
The API attempts to load models in this order:
1. **DistilBERT** (if available) - Highest accuracy
2. **LightGBM** (if available) - Balanced performance  
3. **Logistic Regression** (current) - Fast and reliable

### Training Data Requirements
- CSV format with `text` and `label` columns
- Minimum 2 samples per class
- Supported labels: positive, negative, neutral (case insensitive)
- Automatic text cleaning and preprocessing

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | User-friendly homepage with API guide |
| `/predict` | POST | Single text sentiment prediction |
| `/batch_predict` | POST | Multiple text sentiment predictions |
| `/health` | GET | API health status and model info |
| `/model_info` | GET | Detailed model information |
| `/test` | GET | Simple test endpoint |
| `/docs` | GET | Interactive API documentation |

## Configuration

### Server Settings
The API runs on `http://127.0.0.1:8000` by default for Windows compatibility.

### Input Validation
- Text length: Maximum 1000 characters
- Batch size: Maximum 100 texts per request
- Empty text validation with helpful error messages

### Error Handling
- 400: Bad Request (invalid input)
- 404: Endpoint not found  
- 422: Unprocessable Entity (validation error)
- 500: Internal Server Error (model/prediction failure)

## Troubleshooting

### Common Issues

**1. "Model directory not found"**
```
Solution: Ensure the best_model/ folder exists with all required .pkl files
```

**2. "Label encoder not found"**
```
Solution: Verify label_encoder.pkl exists in best_model/ directory
```

**3. "Cannot access http://0.0.0.0:8000"**
```
Solution: Use http://localhost:8000 instead - never use 0.0.0.0 in browser
```

**4. Scikit-learn version warnings**
```
Solution: Warnings are suppressed in the code and don't affect functionality
```

**5. Port already in use**
```bash
# Find and kill process using port 8000
netstat -ano | findstr :8000
# Then kill the process ID shown
```

### Debug Mode

To enable detailed logging, modify the last line in `app.py`:
```python
uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug", reload=True)
```

## Development

### Adding New Models
To add support for additional models:

1. Place model files in `best_model/` directory
2. Update the `load_models()` function in `app.py`
3. Add prediction logic in `predict_sentiment()` function
4. Test thoroughly with sample data

### Model Training
Use the provided `pipeline.ipynb` notebook to:
- Load and preprocess your dataset
- Train multiple model types
- Compare model performance
- Export the best model to `best_model/` directory

## Testing

### Basic API Test
```bash
# Test if API is running
curl http://localhost:8000/test

# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing product!"}'
```

### Expected Success Output
When starting the server, you should see:
```
🚀 SvaraAI Reply Classification API Server Starting...
📱 IMPORTANT: Use these URLs in your browser:
   🌐 Main API:          http://localhost:8000/
   📚 Documentation:     http://localhost:8000/docs
   🔧 Health Check:      http://localhost:8000/health

INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:__main__:Using device: cpu
INFO:__main__:Loaded label encoder with classes: ['negative' 'neutral' 'positive']
INFO:__main__:✅ Loaded Logistic Regression model
INFO:__main__:🚀 Server started successfully with Logistic Regression model
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

## Requirements

### Python Dependencies
```txt
fastapi>=0.104.0
uvicorn>=0.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
pandas>=2.0.0
pydantic>=2.0.0
```

### System Requirements
- Python 3.8+
- Windows 10/11 (tested)
- 4GB RAM minimum
- CPU-based inference (GPU optional)

## License

This project is available under the MIT License.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review API documentation at `/docs` when running
- Ensure all model files are present in `best_model/` directory
- Verify your dataset format matches requirements

---

**Ready to classify sentiment!** Start the server with `python app.py` and visit http://localhost:8000 to begin.