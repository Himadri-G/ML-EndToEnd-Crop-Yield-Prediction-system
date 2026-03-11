from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Crop Yield Prediction API",
    description="Predicts crop yield (hg/ha) based on agricultural and climate features.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
pipeline = joblib.load("artifacts/model_trainer/pipeline.pkl")

 
try:
    training_columns = pipeline.feature_names_in_         
except AttributeError:
    training_columns = pipeline[-1].feature_names_in_    


# ── Pydantic input schema ─────────────────────────────────────────────────────

class CropInput(BaseModel):
    Area: str = Field(..., example="Albania", description="Country or region name")
    Item: str = Field(..., example="Maize", description="Crop type")
    Year: int = Field(..., example=1990, ge=1960, le=2100, description="Year of cultivation")
    average_rain_fall_mm_per_year: float = Field(..., example=1485.0, ge=0, description="Average annual rainfall in mm")
    pesticides_tonnes: float = Field(..., example=121.0, ge=0, description="Pesticides used in tonnes")
    avg_temp: float = Field(..., example=16.37, description="Average temperature in Celsius")

    class Config:
        json_schema_extra = {
            "example": {
                "Area": "Albania",
                "Item": "Maize",
                "Year": 1990,
                "average_rain_fall_mm_per_year": 1485.0,
                "pesticides_tonnes": 121.0,
                "avg_temp": 16.37
            }
        }


# ── Response schema ───────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    crop: str
    area: str
    year: int
    predicted_yield_hg_per_ha: float
    predicted_yield_tonnes_per_ha: float


# ── One-hot encode input to match training columns ────────────────────────────

def encode_input(data: CropInput) -> pd.DataFrame:
    """
    Manually one-hot encode Area and Item to match the columns
    seen during model training (e.g. Area_Albania, Item_Maize).
    """
   
    row = {
        "Year": data.Year,
        "average_rain_fall_mm_per_year": data.average_rain_fall_mm_per_year,
        "pesticides_tonnes": data.pesticides_tonnes,
        "avg_temp": data.avg_temp,
    }

    
    area_col = f"Area_{data.Area}"
    item_col = f"Item_{data.Item}"

    if area_col not in training_columns:
        valid = [c.replace("Area_", "") for c in training_columns if c.startswith("Area_")]
        raise ValueError(f"Unknown Area: '{data.Area}'. Valid options: {sorted(valid)}")

    if item_col not in training_columns:
        valid = [c.replace("Item_", "") for c in training_columns if c.startswith("Item_")]
        raise ValueError(f"Unknown crop Item: '{data.Item}'. Valid options: {sorted(valid)}")

    row[area_col] = 1
    row[item_col] = 1

  
    input_df = pd.DataFrame([row])
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    return input_df


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Crop Yield Prediction API is running!",
        "docs": "/docs",
        "predict_endpoint": "/predict"
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.get("/valid-inputs", tags=["Info"])
def valid_inputs():
    """Returns all valid Area and Item values the model was trained on."""
    areas = sorted([c.replace("Area_", "") for c in training_columns if c.startswith("Area_")])
    items = sorted([c.replace("Item_", "") for c in training_columns if c.startswith("Item_")])
    return {"valid_areas": areas, "valid_items": items}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: CropInput):
    try:
        input_df = encode_input(data)
        prediction = float(pipeline.predict(input_df)[0])

        return PredictionResponse(
            crop=data.Item,
            area=data.Area,
            year=data.Year,
            predicted_yield_hg_per_ha=round(prediction, 2),
            predicted_yield_tonnes_per_ha=round(prediction / 10000, 4)
        )

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(data: list[CropInput]):
    """Predict yield for multiple crops at once."""
    try:
        results = []
        for d in data:
            input_df = encode_input(d)
            prediction = float(pipeline.predict(input_df)[0])
            results.append({
                "crop": d.Item,
                "area": d.Area,
                "year": d.Year,
                "predicted_yield_hg_per_ha": round(prediction, 2),
                "predicted_yield_tonnes_per_ha": round(prediction / 10000, 4)
            })

        return {"predictions": results, "count": len(results)}

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")
