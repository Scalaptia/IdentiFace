from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from keras.models import load_model
from Backend.functions import Functions

app = FastAPI()

# Load the pre-trained model
model = load_model("Models/shape.h5")

@app.post("/predict/")
async def predict_face_shape(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_data = await file.read()
        image_array = np.fromstring(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Preprocess the image
        path, preprocessed_image = Functions.preprocess("offline", image)

        # Make predictions using the loaded model
        predictions = model.predict(preprocessed_image)

        # Determine the predicted class based on the threshold
        predicted_class_index = np.argmax(predictions)
        predicted_class = None

        if predicted_class_index == 0:
            predicted_class = 'Oblong'
        elif predicted_class_index == 1:
            predicted_class = 'Square'
        elif predicted_class_index == 2:
            predicted_class = 'Round'
        elif predicted_class_index == 3:
            predicted_class = 'Heart'
        elif predicted_class_index == 4:
            predicted_class = 'Oval'

        return JSONResponse(content={"shape": predicted_class})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)