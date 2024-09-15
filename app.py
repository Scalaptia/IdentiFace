from fastapi import FastAPI, File, UploadFile, HTTPException, Query # fastapi, uvicorn, python-multipart
from fastapi.responses import JSONResponse
import cv2 # opencv-python
import numpy as np # numpy
from keras.models import load_model # keras
from Backend.functions import Functions
import requests
from io import BytesIO

app = FastAPI()

# Load the pre-trained models
shape_model = load_model("Models/shape.h5")
gender_model = load_model("Models/gender.h5")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_data = await file.read()
        image_array = np.fromstring(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Preprocess the image
        preprocessed_shape_image, preprocessed_gender_image = Functions.preprocess(image)

        # Make predictions using the loaded models
        shape_predictions = Functions.predict_shape(preprocessed_shape_image, shape_model)
        gender_predictions = Functions.predict_gender(preprocessed_gender_image, gender_model)

        # Extract skin tone palette
        skin_tone_palette = Functions.extract_skin_tone(image)

        return JSONResponse(
            content={
                "forma": shape_predictions[0],
                "genero": gender_predictions[0],
                "tono_piel": skin_tone_palette,
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.post("/predict-url/")
async def predict_url(image: str = Query(...)):
    try:
        # Download the image from the URL
        response = requests.get(image)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Unable to download image")

        image_data = BytesIO(response.content).read()
        image_array = np.fromstring(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Preprocess the image
        preprocessed_shape_image, preprocessed_gender_image = Functions.preprocess(image)

        # Make predictions using the loaded models
        shape_predictions = Functions.predict_shape(preprocessed_shape_image, shape_model)
        gender_predictions = Functions.predict_gender(preprocessed_gender_image, gender_model)

        # Extract skin tone palette
        skin_tone_palette = Functions.extract_skin_tone(image)

        return JSONResponse(
            content={
                "forma": shape_predictions[0],
                "genero": gender_predictions[0],
                "tono_piel": skin_tone_palette,
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)