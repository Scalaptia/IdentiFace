import cv2
import tensorflow as tf
import numpy as np
import dlib
import matplotlib.pyplot as plt

class Functions():
    @staticmethod
    def preprocess(input_image, target_size=(128, 128)):
        "Function to preprocess the extracted faces"
        # Initialize the face detector and landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("Utilities/Face-Detection/shape_predictor_68_face_landmarks.dat")
        
        # Check if the input image is a file path or an image array
        if isinstance(input_image, str):
            # Read the original image from file path
            img = cv2.imread(input_image)
        else:
            # Use the provided image array
            img = input_image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Detect faces in the image
        faces = detector(img)

        # Process the first detected face
        if faces:
            # Get the first face from the list of detected faces
            face = faces[0]

            # Draw a rectangle around the face
            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

            # Extract the face region
            extracted_face = img[face.top():face.bottom(), face.left():face.right()]

            # Check if the extracted face is not empty
            if not extracted_face.size:
                return None

            # Resize the face to the target size
            resized_face = cv2.resize(extracted_face, target_size)
            path = 'test.jpg'
            # Save the resized face as 'test.jpg'
            cv2.imwrite(path, resized_face)

            # Normalize the pixel values to the range [0, 1]
            normalized_face = resized_face / 255.0

            # Expand the dimensions to match the input shape expected by the model
            normalized_face = np.expand_dims(normalized_face, axis=0)

            return path, normalized_face

        # If no faces are found, return None
        return None

    "Shape prediciton Function"
    def predict_shape(preprocessed_image, model):
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

        # Return the predictions
        print(predictions)
        return predicted_class, predictions
    
    
    "Gender Classification Function"
    def predict_gender(preprocessed_image, model):
        # Make predictions using the loaded model
        predictions = model.predict(preprocessed_image)

        # Get the index of the highest element in the predictions array
        predicted_index = np.argmax(predictions)

        # Determine the predicted class based on the index
        predicted_class = 'Male' if predicted_index == 1 else 'Female'
        # Return the predictions
        print(predictions)
        
        return predicted_class, predictions 