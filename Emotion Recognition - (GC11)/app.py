import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool1D, Dropout
from tensorflow.keras.models import Model
import numpy as np

# ---- Load Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained('./bert_tokenizer', use_fast=True)

# ---- Build Model ----
def build_model():
    max_len = 40
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

    # Load pretrained BERT model directly
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    x = GlobalMaxPool1D()(bert_output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(6, activation='softmax')(x)

    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    return model

# ---- Load Model and Weights ----
model = build_model()
model.load_weights('emotion_detector.weights.h5')

# ---- Emotion Labels ----
id2label = {
    0: 'anger',
    1: 'fear',
    2: 'joy',
    3: 'love',
    4: 'sadness',
    5: 'surprise'
}

# ---- Streamlit UI ----
st.title("💬 Emotion Detection with BERT")
st.markdown("Enter a sentence and predict the underlying **emotion** using a fine-tuned BERT model.")

user_input = st.text_area("Input Text", "I'm feeling optimistic about the future!")

if st.button("Predict Emotion"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Tokenize input
        encoded = tokenizer(
            user_input,
            max_length=40,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )

        # Predict
        preds = model.predict([encoded['input_ids'], encoded['attention_mask']])
        pred_class = int(np.argmax(preds))
        probs = preds[0]

        # Show result
        st.success(f"**Predicted Emotion:** {id2label[pred_class].capitalize()}")
        st.write("**Confidence Scores:**")
        st.json({id2label[i]: float(probs[i]) for i in range(6)})
