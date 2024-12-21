from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from datetime import datetime
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import load_model
from io import BytesIO
from tensorflow import keras
import tensorflowjs as tfjs
import os

app = FastAPI()

# Cloud storage URL for the model JSON file
MODEL_JSON_URL = os.getenv('BUCKET_URL')

# Load the model from cloud storage
def load_tfjs_model_from_cloud(json_url):
    """
    Load a TensorFlow.js model stored in cloud storage.

    Args:
        json_url (str): URL of the model.json file.

    Returns:
        keras.Model: The loaded model.
    """
    try:
        # Download the model.json file and weights
        model_json_path = get_file("model.json", origin=json_url)
        model = tfjs.converters.load_keras_model(model_json_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model from {json_url}: {str(e)}")

# Load the model once at startup
try:
    model = load_tfjs_model_from_cloud(MODEL_JSON_URL)
    print(f"Model loaded successfully! Input shape: {model.input_shape}")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

@app.post("/predict")
async def predict_endpoint(file: UploadFile):
    """
    Handle prediction requests by receiving an uploaded image,
    processing it, and returning the prediction result.

    Args:
        file: Uploaded file via multipart/form-data.

    Returns:
        JSONResponse: Response containing prediction results.
    """
    if model is None:
        return JSONResponse(
            content={"status": "fail", "message": "Model not loaded."},
            status_code=500,
        )
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Check file size (max 1MB)
        contents = await file.read()
        if len(contents) > 1000000:
            return JSONResponse(
                content={
                    "status": "fail",
                    "message": "Payload content length greater than maximum allowed: 1000000"
                },
                status_code=413,
            )

        # Process the image
        img = keras.preprocessing.image.load_img(BytesIO(contents), target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Perform prediction
        result = model.predict(img_array)
        prediction = result[0][0]  # Assuming binary classification

        # Return structured responses
        if prediction > 0.5:
            return {
                "status": "success",
                "message": "Model is predicted successfully",
                "data": {
                    "id": "77bd90fc-c126-4ceb-828d-f048dddff746",
                    "result": "Cancer",
                    "suggestion": "Segera periksa ke dokter!",
                    "createdAt": datetime.now().isoformat(),
                },
            }
        else:
            return {
                "status": "success",
                "message": "Model is predicted successfully",
                "data": {
                    "id": "77bd90fc-c126-4ceb-828d-f048dddff746",
                    "result": "Non-cancer",
                    "suggestion": "Penyakit kanker tidak terdeteksi.",
                    "createdAt": datetime.now().isoformat(),
                },
            }

    except Exception as e:
        return JSONResponse(
            content={
                "status": "fail",
                "message": "Terjadi kesalahan dalam melakukan prediksi",
                "error": str(e),
            },
            status_code=400,
        )
