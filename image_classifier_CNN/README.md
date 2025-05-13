# Image Classifier

This project is a web-based application that allows users to upload an image of an electronic component images (resistors, ICs, capacitors) and get the classification result. The backend is built using Flask, and the frontend consists of a simple HTML interface with JavaScript for image uploading and previewing. The trained deep learning model is used to classify the images.

## Features

- **Upload Image**: Users can upload images of electronic components.
- **Image Preview**: Preview the uploaded image before submitting it for classification.
- **Classification**: The model classifies the image and returns the result with a confidence score.

## Instructions for Local Deployment

Follow the steps below to run the application on your local machine.

### 1. Setup Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (MacOS/Linux)
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Make sure the trained deep learning model(``.keras`` file) is located in the ``models/`` directory
### 3. Run the application
```bash
python app.py
```

You should see output indicating that the server is running:
```bash
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```