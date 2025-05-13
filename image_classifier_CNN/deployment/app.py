from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, make_response
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os, shutil
from werkzeug.utils import secure_filename
from db_setup import SessionLocal, ClassifiedImage, TrainingImage
import sqlite3
import zipfile
import io
import csv

# Initialize the Flask app
app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the trained model
# The pretrained model is used, but the other model can also be used
model = load_model('models/best_model_pretrained.keras')

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(640, 640, 3))  
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image and predict
        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)
        
        # three classes: ['Capacitor', 'IC', 'Resistor']
        classes = ['Capacitor', 'IC', 'Resistor']
        predicted_class = classes[np.argmax(prediction)]
        
        # Save metadata to database
        db = SessionLocal()
        image_record = ClassifiedImage(
            filename=filename,
            file_path=filepath,
            predicted_class=predicted_class,
            confidence=float(np.max(prediction))
        )
        db.add(image_record)
        db.commit()
        db.close()

        # Return the result as JSON
        return jsonify({
            'class': predicted_class,
            'confidence': float(np.max(prediction))
        })
    else:
        return redirect(request.url)
    

@app.route('/retrain_process')
def retrain_process():

    session = SessionLocal()
    images = session.query(TrainingImage).all()

    base_dir = 'retrain_data'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # Clear and recreate folders
    for folder in [train_dir, test_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # Save images by class
    for i, img in enumerate(images):
        label_dir = os.path.join(train_dir if i % 5 != 0 else test_dir, img.label)
        os.makedirs(label_dir, exist_ok=True)
        full_path = os.path.join('static', img.filepath.split('static/')[-1])
        try:
            Image.open(full_path).save(os.path.join(label_dir, img.filename))
        except:
            continue

    img_height = img_width = 640
    batch_size = 32

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Define model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model_path = 'models/best_model_pretrained.keras'
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=25, callbacks=[early_stopping, checkpoint])

    # 3. Evaluate
    predictions = model.predict(test_ds)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    cm_img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()

    return render_template('retrain_result.html', accuracy=accuracy, f1=f1, cm_img=cm_img)

@app.route('/retrain_model')
def retrain_model():
    return render_template('retrain_status.html')


@app.route('/history')
def history():
    db = SessionLocal()
    images = db.query(ClassifiedImage).all()
    db.close()
    return render_template('history.html', images=images)


@app.route('/download_history')
def download_history():
    conn = sqlite3.connect('component_images.db')
    c = conn.cursor()
    c.execute('SELECT filename, predicted_class, confidence FROM classified_images')
    rows = c.fetchall()
    conn.close()

    # Create CSV content using StringIO
    from io import StringIO
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Filename', 'Predicted Class', 'Confidence'])

    for row in rows:
        writer.writerow([row[0], row[1], f"{row[2] * 100:.2f}%"])

    # Set the response with CSV content
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=classification_history.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response


@app.route('/dataset')
def dataset():
    db = SessionLocal()
    images = db.query(TrainingImage).all()
    db.close()
    return render_template('dataset.html', images=images)


@app.route('/download_dataset')
def download_dataset():
    db = SessionLocal()
    images = db.query(TrainingImage).all()
    db.close()

    # Create an in-memory zip archive
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for img in images:
            # Create subfolder inside the zip based on the label
            arcname = f"{img.label}/{img.filename}"  # eg. 'Capacitor/image1.jpg'
            zf.write(img.filepath, arcname=arcname)

    memory_file.seek(0)

    return send_file(memory_file, download_name="training_dataset.zip", as_attachment=True)

@app.route('/upload_training_images', methods=['POST'])
def upload_training_images():
    db_session = SessionLocal()
    files = request.files.getlist('files')
    label = request.form['label']
    save_dir = os.path.join('static', 'training_data', label)
    os.makedirs(save_dir, exist_ok=True)

    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(save_dir, filename)
        file.save(filepath)

        new_image = TrainingImage(filename=filename, filepath=filepath, label=label)
        db_session.add(new_image)

    db_session.commit()
    return redirect(url_for('dataset'))


if __name__ == "__main__":
    app.run(debug=True)
