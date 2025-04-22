import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc}")

predictions = model.predict(x_test)

# Visualizing few predictions
for i in range(5):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}")
    plt.show()

model.save("digit_model.h5")





FILE NAME: Desktop.py



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
