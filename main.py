from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import gradio as gr

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("house_model.pkl")

# Define input schema for FastAPI
class Input(BaseModel):
    data: list

# POST endpoint for prediction
@app.post("/predict")
def predict(input: Input):
    prediction = model.predict([input.data])
    return {"prediction": prediction[0]}

# Gradio function
def gradio_predict(val1, val2, val3, val4, val5, val6, val7, val8):
    return model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])[0]

# Gradio interface
demo = gr.Interface(
    fn=gradio_predict,
    inputs=["number"] * 8,
    outputs="number",
    title="🏠 House Price Predictor",
    description="Enter 8 housing features to predict median house price."
)

# Launch the Gradio app if running locally
if __name__ == "__main__":
    demo.launch()
