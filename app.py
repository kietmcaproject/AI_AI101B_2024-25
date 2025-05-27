from flask import Flask, render_template, request, send_from_directory, url_for
import os
from ultralytics import YOLO
import cv2
import uuid

app = Flask(__name__)

# Set up folders
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO('yolov5s.pt')  # make sure this exists

@app.route('/', methods=['GET', 'POST'])
def index():
    result_img = None
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filename = f"{uuid.uuid4().hex}.jpg"
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            result_path = os.path.join(RESULT_FOLDER, filename)

            # Save the uploaded image
            image.save(image_path)

            # Run YOLOv5 on the image
            results = model(image_path)
            # Save the result image
            results[0].save(filename=result_path)

            # Pass relative path to template
            result_img = url_for('static', filename=f"results/{filename}")
    return render_template('index.html', result_img=result_img)

if __name__ == '__main__':
    app.run(debug=True)
