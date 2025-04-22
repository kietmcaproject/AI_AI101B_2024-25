
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0


x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\n🎯 Test Accuracy: {test_accuracy:.4f}")


model.save("digit_recognition_model.h5")


predictions = model.predict(x_test)


shown_digits = set()
plt.figure(figsize=(15, 5))

i = 0
count = 0
while len(shown_digits) < 10 and i < len(x_test):
    label = y_test[i]
    if label not in shown_digits:
        plt.subplot(2, 5, count + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Digit: {label} | Predicted: {np.argmax(predictions[i])}")
        plt.axis('off')
        shown_digits.add(label)
        count += 1
    i += 1

plt.suptitle("Sample of each digit from 0 to 9")
plt.tight_layout()
plt.show()