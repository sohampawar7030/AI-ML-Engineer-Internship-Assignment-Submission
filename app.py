# app.py  
# SvaraAI Reply Classification FastAPI Service - Windows Compatible

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, field_validator
import joblib
import torch
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import logging
from typing import Optional
import uvicorn
import warnings

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with lifespan
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        load_models()
        logger.info(f" Server started successfully with {model_type} model")
        yield
    except Exception as e:
        logger.error(f" Error loading models: {e}")
        raise RuntimeError(f"Failed to load models: {e}")
    # Shutdown
    logger.info("Server shutting down...")

app = FastAPI(
    title="SvaraAI Reply Classification API",
    description="Sentiment classification API for reply text analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Global variables for loaded models
model = None
tokenizer = None
tfidf = None
ml_model = None
label_encoder = None
model_type = None
device = None

# Pydantic models for API
class TextRequest(BaseModel):
    text: str
    
    @field_validator('text')
    @classmethod
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        if len(v) > 1000:
            raise ValueError('Text too long (max 1000 characters)')
        return v.strip()

class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float
    model_used: str

class BatchTextRequest(BaseModel):
    texts: list[str]
    
    @field_validator('texts')
    @classmethod
    def texts_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('Text list cannot be empty')
        if len(v) > 100:
            raise ValueError('Too many texts (max 100 per batch)')
        return v

class HealthResponse(BaseModel):
    status: str
    model_type: Optional[str]
    classes: Optional[list[str]]
    device: Optional[str]

# Text preprocessing function
def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    if not text:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)          # Remove emails
    text = re.sub(r'\s+', ' ', text).strip()     # Remove extra spaces
    text = re.sub(r'[^\w\s!?.,]', '', text)      # Keep basic punctuation
    
    return text

# Model loading functions
def load_models():
    """Load the best model based on what's available"""
    global model, tokenizer, tfidf, ml_model, label_encoder, model_type, device
    
    model_dir = './best_model'
    
    if not os.path.exists(model_dir):
        raise RuntimeError(f"Model directory '{model_dir}' not found. Please train models first.")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load label encoder (always needed)
    label_encoder_path = f'{model_dir}/label_encoder.pkl'
    if not os.path.exists(label_encoder_path):
        raise RuntimeError("Label encoder not found in model directory")
    
    try:
        label_encoder = joblib.load(label_encoder_path)
        logger.info(f"Loaded label encoder with classes: {label_encoder.classes_}")
    except Exception as e:
        raise RuntimeError(f"Could not load label encoder: {e}")
    
    # Try loading models in order of preference: DistilBERT > LightGBM > LogisticRegression
    
    # 1. Try loading DistilBERT first
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model.to(device)
        model.eval()  # Set to evaluation mode
        model_type = "DistilBERT"
        logger.info(" Loaded DistilBERT model")
        return
    except Exception as e:
        logger.info(f"DistilBERT model not found: {e}")
    
    # 2. Try loading LightGBM
    lgb_path = f'{model_dir}/lgb_model.pkl'
    tfidf_path = f'{model_dir}/tfidf_vectorizer.pkl'
    
    if os.path.exists(lgb_path) and os.path.exists(tfidf_path):
        try:
            ml_model = joblib.load(lgb_path)
            tfidf = joblib.load(tfidf_path)
            model_type = "LightGBM"
            logger.info(" Loaded LightGBM model")
            return
        except Exception as e:
            logger.info(f"LightGBM model loading failed: {e}")
    
    # 3. Try loading Logistic Regression
    lr_path = f'{model_dir}/lr_model.pkl'
    
    if os.path.exists(lr_path) and os.path.exists(tfidf_path):
        try:
            ml_model = joblib.load(lr_path)
            if not tfidf:  # Load tfidf if not already loaded
                tfidf = joblib.load(tfidf_path)
            model_type = "Logistic Regression"
            logger.info(" Loaded Logistic Regression model")
            return
        except Exception as e:
            logger.info(f"Logistic Regression model loading failed: {e}")
    
    raise RuntimeError("No valid models found in the model directory")

def predict_sentiment(text: str) -> tuple[int, float]:
    """Predict sentiment for given text"""
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        raise ValueError("Text contains no valid content after cleaning")
    
    if model_type == "DistilBERT":
        # DistilBERT prediction
        inputs = tokenizer(
            cleaned_text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            confidence = probabilities[0][prediction].item()
    
    else:
        # ML model prediction (LightGBM or Logistic Regression)
        text_tfidf = tfidf.transform([cleaned_text])
        prediction = ml_model.predict(text_tfidf)[0]
        
        # Get prediction probabilities for confidence
        if hasattr(ml_model, 'predict_proba'):
            probabilities = ml_model.predict_proba(text_tfidf)[0]
            confidence = float(probabilities[prediction])
        else:
            # Fallback for models without predict_proba
            confidence = 0.8  # Default high confidence
    
    return prediction, confidence

# API Endpoints
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with user-friendly HTML interface"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SvaraAI - Sentiment Analysis API</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                line-height: 1.6;
            }}
            .container {{
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 3px solid #667eea;
            }}
            .status {{
                background: #d4edda;
                color: #155724;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 5px solid #28a745;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 5px solid #667eea;
            }}
            .endpoint {{
                background: white;
                margin: 10px 0;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }}
            .method {{
                background: #007bff;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }}
            .method.post {{ background: #28a745; }}
            .button {{
                display: inline-block;
                background: #667eea;
                color: white;
                padding: 12px 20px;
                text-decoration: none;
                border-radius: 6px;
                margin: 5px;
                transition: all 0.3s;
            }}
            .button:hover {{
                background: #5a67d8;
                transform: translateY(-2px);
            }}
            .example {{
                background: #f1f3f4;
                padding: 15px;
                border-radius: 6px;
                font-family: 'Courier New', monospace;
                margin: 10px 0;
                overflow-x: auto;
            }}
            .model-info {{
                background: #e3f2fd;
                padding: 15px;
                border-radius: 8px;
                border-left: 5px solid #2196f3;
            }}
            h1 {{ color: #667eea; margin: 0; }}
            h2 {{ color: #555; }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e9ecef;
                color: #6c757d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>SvaraAI Sentiment Analysis API</h1>
                <p>Real-time text sentiment classification service</p>
            </div>

            <div class="status">
                <strong>Status:</strong> API is running successfully<br>
                <strong>Model:</strong> {model_type}<br>
                <strong>Version:</strong> 1.0.0<br>
                <strong>Device:</strong> {str(device) if device else 'CPU'}
            </div>

            <div class="section">
                <h2>Quick Start</h2>
                <p><strong>New to APIs?</strong> Click the interactive documentation below to test the API instantly:</p>
                <a href="/docs" class="button">Try API Now - Interactive Docs</a>
                <a href="/health" class="button">Check API Health</a>
            </div>

            <div class="section">
                <h2>Available Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method post">POST</span> <strong>/predict</strong> - Analyze single text sentiment
                    <div class="example">
curl -X POST "http://localhost:8000/predict" \\
-H "Content-Type: application/json" \\
-d '{{"text": "This product is amazing!"}}'</div>
                </div>

                <div class="endpoint">
                    <span class="method post">POST</span> <strong>/batch_predict</strong> - Analyze multiple texts
                    <div class="example">
curl -X POST "http://localhost:8000/batch_predict" \\
-H "Content-Type: application/json" \\
-d '{{"texts": ["Great service!", "Terrible experience", "It was okay"]}}'</div>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> <strong>/health</strong> - API health status
                    <div class="example">curl http://localhost:8000/health</div>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> <strong>/model_info</strong> - Model details
                    <div class="example">curl http://localhost:8000/model_info</div>
                </div>
            </div>

            <div class="section">
                <h2>Example Response</h2>
                <div class="example">
{{
  "text": "This product is amazing!",
  "label": "positive",
  "confidence": 0.8542,
  "model_used": "{model_type}"
}}
                </div>
            </div>

            <div class="model-info">
                <h2>Model Information</h2>
                <p><strong>Classes:</strong> {', '.join(label_encoder.classes_.tolist()) if label_encoder else 'Loading...'}</p>
                <p><strong>Model Type:</strong> {model_type}</p>
                <p><strong>Processing:</strong> Text preprocessing with TF-IDF vectorization</p>
            </div>

            <div class="section">
                <h2>Integration Examples</h2>
                <h3>Python</h3>
                <div class="example">
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={{"text": "Your text here"}}
)
result = response.json()
print(f"Sentiment: {{result['label']}} ({{result['confidence']:.2f}})")
                </div>

                <h3>JavaScript</h3>
                <div class="example">
fetch('http://localhost:8000/predict', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{'text': 'Your text here'}})
}})
.then(response => response.json())
.then(data => console.log('Sentiment:', data.label, data.confidence));
                </div>
            </div>

            <div class="footer">
                <p>SvaraAI Reply Classification API | Built with FastAPI</p>
                <p>For detailed API documentation and testing: <a href="/docs">Interactive API Docs</a></p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_type=model_type,
        classes=label_encoder.classes_.tolist() if label_encoder else [],
        device=str(device) if device else None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    """
    Predict sentiment for a single text
    
    **Example request:**
    ```json
    {
        "text": "Looking forward to the demo!"
    }
    ```
    
    **Example response:**
    ```json
    {
        "text": "Looking forward to the demo!",
        "label": "positive",
        "confidence": 0.87,
        "model_used": "Logistic Regression"
    }
    ```
    """
    try:
        prediction, confidence = predict_sentiment(request.text)
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        return PredictionResponse(
            text=request.text,
            label=predicted_label,
            confidence=round(confidence, 4),
            model_used=model_type
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(request: BatchTextRequest):
    """
    Predict sentiment for multiple texts
    
    **Example request:**
    ```json
    {
        "texts": [
            "Great service!",
            "Terrible experience",
            "It was okay"
        ]
    }
    ```
    """
    try:
        results = []
        
        for text in request.texts:
            try:
                if text and text.strip():
                    prediction, confidence = predict_sentiment(text)
                    predicted_label = label_encoder.inverse_transform([prediction])[0]
                    
                    results.append({
                        "text": text,
                        "label": predicted_label,
                        "confidence": round(confidence, 4),
                        "model_used": model_type,
                        "status": "success"
                    })
                else:
                    results.append({
                        "text": text,
                        "label": "unknown",
                        "confidence": 0.0,
                        "model_used": model_type,
                        "status": "error",
                        "error": "Empty or invalid text"
                    })
            
            except Exception as e:
                results.append({
                    "text": text,
                    "label": "unknown",
                    "confidence": 0.0,
                    "model_used": model_type,
                    "status": "error",
                    "error": str(e)
                })
        
        successful_predictions = sum(1 for r in results if r["status"] == "success")
        
        return {
            "predictions": results,
            "total_count": len(results),
            "successful_count": successful_predictions,
            "failed_count": len(results) - successful_predictions,
            "model_used": model_type
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info")
async def model_info():
    """Get detailed model information"""
    return {
        "model_type": model_type,
        "classes": label_encoder.classes_.tolist() if label_encoder else [],
        "num_classes": len(label_encoder.classes_) if label_encoder else 0,
        "device": str(device) if device else None,
        "model_files": {
            "directory": "./best_model",
            "files": os.listdir("./best_model") if os.path.exists("./best_model") else []
        }
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return {
        "status": "API is working!",
        "timestamp": str(torch.cuda.current_device() if torch.cuda.is_available() else "CPU"),
        "message": "If you can see this, your server is running correctly",
        "next_steps": [
            "Try the /predict endpoint",
            "Visit /docs for interactive documentation",
            "Check /health for system status"
        ]
    }

# Exception handlers
from fastapi.responses import JSONResponse

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "Available endpoints: /, /predict, /batch_predict, /health, /model_info, /test, /docs",
            "tip": "Visit /docs for interactive API documentation"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Please check server logs for details",
            "support": "Check the console output for detailed error information"
        }
    )

# Run server - COMPLETELY UPDATED CONFIGURATION
if __name__ == "__main__":
    print("=" * 80)
    print(" SvaraAI Reply Classification API Server Starting...")
    print("=" * 80)
    print()
    print(" IMPORTANT: Use these URLs in your browser:")
    print(f"    Main API:          http://localhost:8000/")
    print(f"    Documentation:     http://localhost:8000/docs")
    print(f"    Health Check:      http://localhost:8000/health")
    print(f"    Model Info:        http://localhost:8000/model_info")
    print(f"    Test Endpoint:     http://localhost:8000/test")
    print()
    print("  DO NOT use http://0.0.0.0:8000 - it won't work!")
    print("  ALWAYS use http://localhost:8000 instead")
    print()
    print(" Quick Test Commands:")
    print('   curl http://localhost:8000/test')
    print('   curl http://localhost:8000/health')
    print()
    print("=" * 80)
    
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",  # Changed to 127.0.0.1 for better Windows compatibility
            port=8000,
            reload=False,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f" Failed to start server: {e}")
        print(" Try running on a different port:")
        print("   python app.py --port 8001")
        input("Press Enter to exit...")