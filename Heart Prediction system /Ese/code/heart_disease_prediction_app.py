import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import streamlit as st

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('heart.csv')
    return data

data = load_data()

st.title("❤️ Heart Disease Prediction Using ANN")

st.write("### Dataset Preview")
st.dataframe(data.head())

if st.checkbox("Show data summary"):
    st.write(data.describe())

if st.checkbox("Show class distribution"):
    st.bar_chart(data['target'].value_counts())

# Separate features and target
X = data.drop('target', axis=1).values
y = data['target'].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build ANN model
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

st.write("### Training the model... This may take a moment.")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=0
)

st.success("Training complete!")

# Plot training accuracy and loss
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history['accuracy'], label='Training Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[0].set_title('Accuracy')
ax[0].legend()

ax[1].plot(history.history['loss'], label='Training Loss', color='orange')
ax[1].plot(history.history['val_loss'], label='Validation Loss', color='red')
ax[1].set_title('Loss')
ax[1].legend()

st.pyplot(fig)

# Evaluate model on test data
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.write("### Model Evaluation on Test Set")
st.write(f"Accuracy: {accuracy*100:.2f}%")
st.write(f"Precision: {precision*100:.2f}%")
st.write(f"Recall: {recall*100:.2f}%")
st.write(f"F1 Score: {f1*100:.2f}%")
st.write("Confusion Matrix:")
st.write(cm)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Receiver Operating Characteristic')
ax_roc.legend(loc="lower right")

st.pyplot(fig_roc)

# User input for prediction
st.write("### Predict Heart Disease for New Patient Data")

def user_input_features():
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex (1 = male, 0 = female)', [1, 0])
    cp = st.selectbox('Chest Pain Type (0-3)', [0,1,2,3])
    trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
    chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=240)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', [1, 0])
    restecg = st.selectbox('Resting Electrocardiographic Results (0-2)', [0,1,2])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina (1 = yes; 0 = no)', [1, 0])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
    slope = st.selectbox('Slope of the Peak Exercise ST Segment (0-2)', [0,1,2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (0-4)', [0,1,2,3,4])
    thal = st.selectbox('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)', [1, 2, 3])

    features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    return features

input_data = user_input_features()
input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction_prob = model.predict(input_data_scaled)[0][0]
    prediction = 1 if prediction_prob > 0.5 else 0

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"The model predicts that the patient **has heart disease** with a probability of {prediction_prob:.2f}.")
    else:
        st.success(f"The model predicts that the patient **does NOT have heart disease** with a probability of {1 - prediction_prob:.2f}.")
