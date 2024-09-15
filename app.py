from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from keras.models import load_model
from Backend.functions import Functions

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
                "shape": shape_predictions[0],
                "gender": gender_predictions[0],
                "skin_tone_palette": skin_tone_palette,
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)