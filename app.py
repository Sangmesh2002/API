from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
import joblib

# Define FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("models.pkl")

# Define input data schema using Pydantic BaseModel
class InputData(BaseModel):
    state: int
    district: int
    commodity: int
    month: int
    season: int


@app.post("/predict")
async def predict_price(data: InputData):
  
    input_data = [[data.state, data.district, data.commodity, data.month,data.season]]
    
    prediction = model.predict(input_data)
    # Return prediction as response
    return {"predicted_price": prediction[0]}

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
