import cv2
import numpy as np
from flask import Flask, request, render_template
import logging

# ... (previous code)
# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the model architecture from the JSON file
model_name = "model_LicensePlate"
json_file_path = f"{model_name}.json"
h5_file_path = f"{model_name}.h5"

try:
    json_file = open(json_file_path, "r")
    model_json = json_file.read()
    json_file.close()

    # Assuming you have a function to load your model
    # Adjust this part according to your model loading method
    model = load_model_from_json(model_json)

    # Load the model weights from the HDF5 file
    model.load_weights(h5_file_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


@app.route("/", methods=["GET", "POST"])
def predict_license_plate():
    try:
        if request.method == "POST":
            # Get the uploaded image from the POST request
            uploaded_file = request.files["file"]

            if uploaded_file.filename != "":
                # Save the uploaded image to a temporary file
                temp_image_path = "temp_image.jpg"
                uploaded_file.save(temp_image_path)

                if model is not None:
                    # Preprocess the uploaded image (similar to what you did during training)
                    preprocessed_image = preprocess_license_plate(temp_image_path)

                    # Use the loaded model to predict the license plate number
                    predicted_plate = model.predict(
                        np.expand_dims(preprocessed_image, axis=0)
                    )

                    # Convert the predicted_plate to a readable format (e.g., converting indices to characters)
                    # Construct the license plate number as a string
                    # Example: Convert indices to characters and join them
                    plate_number = "".join([str(char) for char in predicted_plate])

                    # Log the recognized plate number
                    logging.info(f"Recognized Plate Number: {plate_number}")

                    return render_template("result.html", plate_number=plate_number)

        return render_template("upload.html")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return "An error occurred while processing the request."


# Define a function to preprocess the license plate image
def preprocess_license_plate(image_path):
    # Load the image from the provided path
    img_ori = cv2.imread(image_path)
    # Apply preprocessing steps here:
    # 1. Convert the image to grayscale (if it's not already)
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    # 2. Apply any necessary filters or enhancements (e.g., Gaussian blur)
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    # 3. Apply thresholding to binarize the image
    _, img_thresh = cv2.threshold(
        img_blurred,
        thresh=0,
        maxval=255,
        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    # 4. Optionally, perform morphological operations for noise reduction
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, structuringElement)
    # Ensure that you return the preprocessed image or relevant data
    # Convert the image to a 3-channel image (grayscale to BGR)
    img_morph = cv2.cvtColor(img_morph, cv2.COLOR_GRAY2BGR)

    # Resize the image to (28, 28)
    img_morph = cv2.resize(img_morph, (28, 28))

    return img_morph  # You can return the binarized image or any other processed data


if __name__ == "__main__":
    app.run(debug=True)
