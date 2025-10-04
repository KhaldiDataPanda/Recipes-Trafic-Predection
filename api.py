from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import io
import uvicorn
from model_loader import ModelLoader

# Initialize FastAPI app
app = FastAPI(
    title="Recipe Traffic Prediction API",
    description="API for predicting recipe traffic based on nutritional and categorical features",
    version="1.0.0"
)

# Load model, scaler, and encoder at startup using dynamic loader
try:
    loader = ModelLoader(model_dir="model")
    model, scaler, encoder, metadata = loader.load_all()
    CATEGORY_ENCODING = loader.get_category_encoding()
    print("✅ Model, scaler, and encoder loaded successfully!")
    if metadata:
        print(f"   Model: {metadata.get('best_model_name', 'Unknown')}")
        print(f"   Accuracy: {metadata.get('accuracy', 0):.4f}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None
    encoder = None
    CATEGORY_ENCODING = {}

# Pydantic models for request/response
class RecipeInput(BaseModel):
    recipe: str = Field(..., description="Recipe name or ID")
    calories: float = Field(..., ge=0, description="Calorie content")
    carbohydrate: float = Field(..., ge=0, description="Carbohydrate content in grams")
    sugar: float = Field(..., ge=0, description="Sugar content in grams")
    protein: float = Field(..., ge=0, description="Protein content in grams")
    category: str = Field(..., description="Recipe category")
    servings: int = Field(..., ge=1, description="Number of servings")
    
    class Config:
        json_schema_extra = {
            "example": {
                "recipe": "001",
                "calories": 250.5,
                "carbohydrate": 35.2,
                "sugar": 12.5,
                "protein": 15.0,
                "category": "Chicken",
                "servings": 4
            }
        }

class PredictionOutput(BaseModel):
    recipe: str
    predicted_traffic: str
    confidence_low: float
    confidence_high: float
    max_confidence: float

class BatchPredictionOutput(BaseModel):
    total_recipes: int
    predictions: List[PredictionOutput]
    summary: dict

# Preprocessing functions
def preprocess_recipe(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess recipe data following the notebook preprocessing steps
    """
    data = data.copy()
    
    # Drop high_traffic column if it exists (it's the target, not a feature)
    if 'high_traffic' in data.columns:
        data = data.drop('high_traffic', axis=1)
    
    # Validate category values - reject if missing
    if 'category' in data.columns:
        if data['category'].isnull().any():
            raise ValueError("Data contains missing 'category' values. All recipes must have a category.")
        
        # Warn about unknown categories but allow processing (encoder will handle)
        if CATEGORY_ENCODING:
            unknown_categories = set(data['category'].unique()) - set(CATEGORY_ENCODING.keys())
            if unknown_categories:
                print(f"⚠️ Warning: Unknown categories found: {', '.join(unknown_categories)}. Using mean encoding.")
    
    # Handle servings (as done in notebook)
    if 'servings' in data.columns:
        data['servings'] = data['servings'].replace({'4 as a snack': 4, '6 as a snack': 6})
        data['servings'] = pd.to_numeric(data['servings'], errors='coerce').fillna(4).astype(int)
    
    # Handle missing values in nutritional columns (exactly as in notebook)
    for col in ['calories', 'carbohydrate', 'sugar', 'protein']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # Fill missing values with category mean
            data[col] = data.groupby('category')[col].transform(
                lambda x: x.fillna(round(x.mean(), 2))
            )
            # If still any NaN, fill with overall mean
            if data[col].isnull().any():
                data[col] = data[col].fillna(round(data[col].mean(), 2))
    
    return data

def encode_categorical(data: pd.DataFrame, category_encoder, category_encoding_dict: dict) -> pd.DataFrame:
    """
    Encode categorical variable using the fitted encoder from training.
    Falls back to manual mapping if encoder transform fails.
    """
    data = data.copy()
    if 'category' not in data.columns:
        return data
    
    try:
        # Try using the encoder's transform method
        data['category'] = category_encoder.transform(data[['category']])
    except Exception as e:
        # Fallback to manual mapping using the encoding dictionary
        if category_encoding_dict:
            mean_value = np.mean(list(category_encoding_dict.values()))
            data['category'] = data['category'].map(category_encoding_dict).fillna(mean_value)
        else:
            raise ValueError(f"Cannot encode categories: encoder transform failed and no encoding dictionary available. Error: {e}")
    
    return data

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Recipe Traffic Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predict traffic for a single recipe",
            "POST /predict/batch": "Predict traffic for multiple recipes",
            "POST /predict/csv": "Upload CSV file for batch predictions",
            "GET /health": "Check API health status",
            "GET /categories": "Get list of supported categories"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoder_loaded": encoder is not None,
        "categories_count": len(CATEGORY_ENCODING) if CATEGORY_ENCODING else 0
    }

@app.get("/categories")
async def get_categories():
    """Get list of supported recipe categories"""
    return {
        "categories": list(CATEGORY_ENCODING.keys()),
        "total": len(CATEGORY_ENCODING)
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_single(recipe: RecipeInput):
    """
    Predict traffic for a single recipe
    """
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([recipe.dict()])
        
        # Preprocess
        processed_data = preprocess_recipe(input_data)
        
        # Encode category using the encoder
        processed_data = encode_categorical(processed_data, encoder, CATEGORY_ENCODING)
        
        # Store recipe name
        recipe_name = processed_data['recipe'].values[0]
        
        # Prepare features
        X = processed_data.drop('recipe', axis=1)
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        prediction_proba = model.predict_proba(X_scaled)[0]
        
        return PredictionOutput(
            recipe=recipe_name,
            predicted_traffic="High" if prediction == 1 else "Low",
            confidence_low=round(float(prediction_proba[0]), 4),
            confidence_high=round(float(prediction_proba[1]), 4),
            max_confidence=round(float(max(prediction_proba)), 4)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(recipes: List[RecipeInput]):
    """
    Predict traffic for multiple recipes
    """
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([r.dict() for r in recipes])
        
        # Preprocess
        processed_data = preprocess_recipe(input_data)
        
        # Store recipe names
        recipe_names = processed_data['recipe'].values
        
        # Encode category using the encoder
        processed_data = encode_categorical(processed_data, encoder, CATEGORY_ENCODING)
        
        # Prepare features
        X = processed_data.drop('recipe', axis=1)
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Predict
        predictions = model.predict(X_scaled)
        predictions_proba = model.predict_proba(X_scaled)
        
        # Create output
        prediction_outputs = []
        for i, name in enumerate(recipe_names):
            prediction_outputs.append(
                PredictionOutput(
                    recipe=name,
                    predicted_traffic="High" if predictions[i] == 1 else "Low",
                    confidence_low=round(float(predictions_proba[i][0]), 4),
                    confidence_high=round(float(predictions_proba[i][1]), 4),
                    max_confidence=round(float(max(predictions_proba[i])), 4)
                )
            )
        
        # Create summary
        high_count = int(sum(predictions))
        low_count = len(predictions) - high_count
        avg_confidence = float(np.mean([max(p) for p in predictions_proba]))
        
        summary = {
            "total_recipes": len(recipes),
            "high_traffic_count": high_count,
            "low_traffic_count": low_count,
            "high_traffic_percentage": round(high_count / len(recipes) * 100, 2),
            "average_confidence": round(avg_confidence, 4)
        }
        
        return BatchPredictionOutput(
            total_recipes=len(recipes),
            predictions=prediction_outputs,
            summary=summary
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict/csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and get predictions for all recipes
    """
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    
    # Check file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Check required columns
        required_cols = ['recipe', 'calories', 'carbohydrate', 'sugar', 'protein', 'category', 'servings']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_cols)}"
            )
        
        # Preprocess
        processed_data = preprocess_recipe(df)
        
        # Store recipe names
        recipe_names = processed_data['recipe'].values
        
        # Encode category using the encoder
        processed_data = encode_categorical(processed_data, encoder, CATEGORY_ENCODING)
        
        # Prepare features
        X = processed_data.drop('recipe', axis=1)
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Predict
        predictions = model.predict(X_scaled)
        predictions_proba = model.predict_proba(X_scaled)
        
        # Create results
        results = []
        for i, name in enumerate(recipe_names):
            results.append({
                "recipe": name,
                "predicted_traffic": "High" if predictions[i] == 1 else "Low",
                "confidence_low": round(float(predictions_proba[i][0]), 4),
                "confidence_high": round(float(predictions_proba[i][1]), 4),
                "max_confidence": round(float(max(predictions_proba[i])), 4)
            })
        
        # Summary statistics
        high_count = int(sum(predictions))
        low_count = len(predictions) - high_count
        avg_confidence = float(np.mean([max(p) for p in predictions_proba]))
        
        return {
            "filename": file.filename,
            "total_recipes": len(df),
            "predictions": results,
            "summary": {
                "high_traffic_count": high_count,
                "low_traffic_count": low_count,
                "high_traffic_percentage": round(high_count / len(df) * 100, 2),
                "average_confidence": round(avg_confidence, 4)
            }
        }
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
