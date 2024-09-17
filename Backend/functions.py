import cv2 # opencv-python
import numpy as np # numpy
import dlib # dlib
import stone as st # skin-tone-classifier
import os
from webcolors import hex_to_rgb
from .colors import skin_tone_categories, seasons, ColorFunctions

class Functions():
    @staticmethod
    def preprocess(input_image, target_size_shape=(128, 128), target_size_gender=(48, 48)):
        "Function to preprocess the extracted faces"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("Utilities/Face-Detection/shape_predictor_68_face_landmarks.dat")
        
        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        faces = detector(img)

        if faces:
            face = faces[0]
            landmarks = predictor(img, face)
            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            extracted_face = img[face.top():face.bottom(), face.left():face.right()]

            if not extracted_face.size:
                return None

            # Resize for shape model
            resized_face_shape = cv2.resize(extracted_face, target_size_shape)
            normalized_face_shape = resized_face_shape / 255.0
            expanded_face_shape = np.expand_dims(normalized_face_shape, axis=0)

            # Resize for gender model
            resized_face_gender = cv2.resize(extracted_face, target_size_gender)
            normalized_face_gender = resized_face_gender / 255.0
            reshaped_face_gender = np.expand_dims(normalized_face_gender, axis=-1)
            expanded_face_gender = np.expand_dims(reshaped_face_gender, axis=0)

            return expanded_face_shape, expanded_face_gender

        return None

    "Shape prediciton Function"
    @staticmethod
    def predict_shape(preprocessed_image, model):
        # Make predictions using the loaded model
        predictions = model.predict(preprocessed_image)

        # Determine the predicted class based on the threshold
        predicted_class_index = np.argmax(predictions)
        predicted_class = None

        if predicted_class_index == 0:
            # Oblong
            predicted_class = 'Alargada'
        elif predicted_class_index == 1:
            # Square
            predicted_class = 'Cuadrada'
        elif predicted_class_index == 2:
            # Round
            predicted_class = 'Redonda'
        elif predicted_class_index == 3:
            # Heart
            predicted_class = 'Corazón' 
        elif predicted_class_index == 4:
            # Oval
            predicted_class = 'Ovalada'

        return predicted_class, predictions
    
    "Gender Classification Function"
    @staticmethod
    def predict_gender(preprocessed_image, model):
        # Make predictions using the loaded model
        predictions = model.predict(preprocessed_image)

        # Determine the predicted class based on the index
        predicted_index = np.argmax(predictions)
        predicted_class = 'Masculino' if predicted_index == 1 else 'Femenino'
        
        return predicted_class, predictions
    
    "Skin Tone Extraction Function"
    @staticmethod
    def extract_skin_tone(image):
        # Save the image to a temporary file
        image_path = 'temp.jpg'
        cv2.imwrite(image_path, image)

        result = st.process(
            filename_or_url=image_path,
            image_type='color',
            tone_palette=None,
            tone_labels=None,
            n_dominant_colors=5, # Number of dominant colors to extract
            return_report_image=False
        )

        # Remove the temporary image file
        os.remove(image_path)
        
        # Extract and return the skin tone palette from the result
        if 'faces' in result and len(result['faces']) > 0:
            dominant_colors = result['faces'][0].get('dominant_colors', [])
            tone_palette = [color['color'] for color in dominant_colors]
            return tone_palette
            
        return []
    
    @staticmethod
    def closest_skin_tone(skin_hex):
        skin_rgb = hex_to_rgb(skin_hex)
        min_distance = float('inf')
        closest_tone = None

        for tone, tone_hex in skin_tone_categories.items():
            tone_rgb = hex_to_rgb(tone_hex)
            distance = ColorFunctions.color_distance(skin_rgb, tone_rgb)
            if distance < min_distance:
                min_distance = distance
                closest_tone = tone
        
        return closest_tone

    @staticmethod
    def suggest_colors(skin_hex):
        skin_tone = Functions.closest_skin_tone(skin_hex)

        suggestions = {}
        for season, tones in seasons.items():
            if skin_tone in tones:
                suggestions[season] = tones[skin_tone]
        
        return skin_tone, suggestions