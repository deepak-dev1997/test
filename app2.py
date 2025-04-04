import os
import json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
from PIL import Image
from tf_keras.models import load_model
import random

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

# Define paths
MODEL_PATH = os.path.join('saved_models2', 'unet_rooftop.h5')
LABEL_MAPPING_PATH = os.path.join('saved_models2', 'label_mapping.json')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
MASK_FOLDER = os.path.join('static', 'masks')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MASK_FOLDER'] = MASK_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load the trained model
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Load label mapping
with open(LABEL_MAPPING_PATH, 'r') as f:
    label_mapping = json.load(f)

# Create inverse mapping (ID to label)
id_to_label = {int(v): k for k, v in label_mapping.items()}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(512, 512)):
    """
    Load and preprocess the image.
    """
    image = Image.open(image_path).convert("RGBA")  # Ensure 4 channels
    image = image.resize(target_size)
    image = np.array(image)
    image = image.astype(np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_mask(image):
    """
    Predict the segmentation mask for the given image.
    """
    preds = model.predict(image)[0]  # Remove batch dimension
    preds = np.argmax(preds, axis=-1)  # Convert to class labels
    return preds

def create_color_mask(pred_mask):
    """
    Convert the predicted mask to a color mask with unique colors for each instance.
    """
    color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)

    unique_labels = np.unique(pred_mask)
    unique_labels = unique_labels[unique_labels != 0]  # Assuming 0 is background

    for label in unique_labels:
        # Create a binary mask for the current label
        binary_mask = (pred_mask == label).astype(np.uint8)

        # Find connected components (instances) in the binary mask
        num_labels, labels_im = cv2.connectedComponents(binary_mask)

        for instance in range(1, num_labels):  # Start from 1 to skip the background
            # Generate a random color for each instance
            color = [random.randint(0, 255) for _ in range(3)]
            # Assign the color to the corresponding pixels in the color mask
            color_mask[labels_im == instance] = color

    return color_mask

def overlay_masks(original_image_path, color_mask, output_path, alpha=0.5):
    """
    Overlay the color mask on the original image and save the result.
    """
    original = cv2.imread(original_image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)  # Convert to RGB
    original = cv2.resize(original, (color_mask.shape[1], color_mask.shape[0]))

    overlay = cv2.addWeighted(original, alpha, color_mask, 1 - alpha, 0)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))  # Save as BGR
    return output_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser may submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Preprocess the image
            image = preprocess_image(upload_path)

            # Predict the mask
            pred_mask = predict_mask(image)

            # Create a color mask with unique colors for each instance
            color_mask = create_color_mask(pred_mask)

            # Save the color mask
            mask_filename = 'mask_' + filename
            mask_path = os.path.join(app.config['MASK_FOLDER'], mask_filename)
            cv2.imwrite(mask_path, color_mask)

            # Overlay masks on the original image
            overlay_filename = 'overlay_' + filename
            overlay_path = os.path.join(app.config['MASK_FOLDER'], overlay_filename)
            overlay_masks(upload_path, color_mask, overlay_path)

            # Pass the filenames to the result template
            return render_template('result.html', original_image=filename, mask_image=mask_filename, overlay_image=overlay_filename)
        else:
            flash('Allowed file types are png, jpg, jpeg, gif')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/api/masked', methods=['POST'])
def api_masked():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        # Secure and create a unique filename to avoid conflicts
        filename = secure_filename(file.filename)
        unique_filename = f"{random.randint(0, 100000)}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(upload_path)

        # Preprocess the image and predict the mask
        image = preprocess_image(upload_path)
        pred_mask = predict_mask(image)
        color_mask = create_color_mask(pred_mask)

        # Save the color mask to the masks folder
        mask_filename = 'mask_' + unique_filename
        mask_path = os.path.join(app.config['MASK_FOLDER'], mask_filename)
        cv2.imwrite(mask_path, color_mask)

        # Return the masked image file as the API response
        return send_from_directory(app.config['MASK_FOLDER'], mask_filename)
    else:
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg, gif'}), 400



@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/masks/<filename>')
def mask_file(filename):
    return send_from_directory(app.config['MASK_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
