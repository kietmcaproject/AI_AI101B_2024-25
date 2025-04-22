import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model

# Load trained digit recognition model
model = load_model("digit_model.h5")

# Constants
CANVAS_SIZE = 400
IMAGE_SIZE = 28

# Create main window
window = tk.Tk()
window.title("Handwritten Digit Recognizer")
window.geometry("500x600")  # Increased height

# Drawing canvas
canvas = tk.Canvas(window, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white', bd=2, relief='ridge')
canvas.pack(pady=10)

# Create image for drawing
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
draw = ImageDraw.Draw(image)

# Drawing function
def draw_digit(event):
    x, y = event.x, event.y
    r = 10
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
    draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

canvas.bind("<B1-Motion>", draw_digit)

# Prediction function
def predict_digit():
    img_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img_resized)
    img_array = 255 - img_array  # Invert (white background)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    result_label.config(text=f"🧠 Predicted Digit: {predicted_digit}")

# Clear function
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)
    result_label.config(text="Draw a digit and click Predict")

# Buttons
btn_frame = tk.Frame(window)
btn_frame.pack(pady=10)

predict_btn = tk.Button(btn_frame, text="Predict", command=predict_digit, font=("Arial", 14), width=10)
predict_btn.grid(row=0, column=0, padx=10)

clear_btn = tk.Button(btn_frame, text="Clear", command=clear_canvas, font=("Arial", 14), width=10)
clear_btn.grid(row=0, column=1, padx=10)

# Result Label
result_label = tk.Label(window, text="Draw a digit and click Predict", font=("Helvetica", 18), fg="blue")
result_label.pack(pady=20)

# Run GUI loop
window.mainloop()
