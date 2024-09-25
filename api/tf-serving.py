from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

endpoint = 'http://localhost:8501/v1/models/PotatoDisease:predict'

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def bytes_to_image_array(bytes) -> np.ndarray:
    image = np.array(Image.open(BytesIO(bytes)))
    return image

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    bytes = await file.read()
    image = bytes_to_image_array(bytes)
    batch = np.expand_dims(image, axis=0)
    
    response = requests.post(endpoint, json={'instances': batch.tolist()})
    prediction = response.json()["predictions"]

    prediction_tensor = prediction['dense_5']
    prediction_np = prediction_tensor.numpy()
    
    predicted_class = CLASS_NAMES[np.argmax(prediction_np)]
    confidence = round(np.max(prediction_np) * 100, 2)
    
    return {'class': predicted_class, 'confidence': confidence}

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='localhost')
