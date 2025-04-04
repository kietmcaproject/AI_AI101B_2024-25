import os
import random
import pyaudio
import wave
import tkinter as tk
from tkinter import messagebox
from google.cloud import language_v1, speech

# ======== Step 1: Set Up Google Cloud Authentication =========
json_key_filename = "ai-powered-key.json"  
json_key_path = os.path.join(os.getcwd(), json_key_filename)

if not os.path.exists(json_key_path):
    messagebox.showerror("Error", f"Google Cloud key file '{json_key_filename}' not found!")
    exit()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path

# ======== Step 2: Initialize Google Cloud Clients =========
try:
    language_client = language_v1.LanguageServiceClient()
    speech_client = speech.SpeechClient()
except Exception as e:
    messagebox.showerror("Error", f"Failed to initialize Google Cloud clients: {e}")
    exit()

# ======== Step 3: Generate Random Number to Guess =========
actual_number = random.randint(1, 100)
print(f"🔢 Correct Number (for testing/debugging): {actual_number}")  # You can remove this

# ======== Step 4: AI-Powered Hint Generation =========
def generate_ai_hint(guess, actual_number):
    if guess == actual_number:
        return f"🎉 Congratulations! You guessed the correct number: {actual_number}!"
    
    direction = "higher" if guess < actual_number else "lower"
    return f"Try a {direction} number!"

# ======== Step 5: Voice Recording and Speech Recognition =========
def record_audio(filename, duration=3, rate=16000, channels=1):
    """Records an audio file from the microphone."""
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=1024)
        frames = [stream.read(1024) for _ in range(0, int(rate / 1024 * duration))]
        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
    except Exception as e:
        messagebox.showerror("Error", f"Failed to record audio: {e}")

def transcribe_audio(filename):
    """Transcribes an audio file using Google Speech-to-Text API."""
    try:
        with open(filename, 'rb') as audio_file:
            content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        response = speech_client.recognize(config=config, audio=audio)

        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            print(f"🎙️ Detected Speech: {transcript}")  # Debugging
            return transcript
        else:
            return None
    except Exception as e:
        messagebox.showerror("Error", f"Speech recognition failed: {e}")
        return None

def extract_number_from_speech(transcript):
    """Extracts numbers from transcribed text (fixes recognition issues)."""
    words = transcript.split()
    for word in words:
        if word.isdigit():
            return int(word)  # Convert valid numbers
    return None  # If no valid number is found

# ======== Step 6: Handling Voice-Based Number Guessing =========
def voice_guess():
    """Handles voice input and extracts the guessed number."""
    audio_filename = "user_guess.wav"
    record_audio(audio_filename)
    transcription = transcribe_audio(audio_filename)
    if transcription:
        guess = extract_number_from_speech(transcription)
        return guess
    return None

# ======== Step 7: GUI Implementation =========
def check_guess():
    global actual_number
    try:
        guess = int(entry.get())
        hint = generate_ai_hint(guess, actual_number)
        result_label.config(text=hint)

        if guess == actual_number:
            messagebox.showinfo("🎉 Congratulations", f"You guessed the correct number: {actual_number}!\nStarting a new round.")
            actual_number = random.randint(1, 100)  # Restart game
            print(f"🔢 New Correct Number: {actual_number}")  # Debugging
            entry.delete(0, tk.END)  # Clear input
            result_label.config(text="New number generated. Try again!")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number.")

def voice_guess_handler():
    """Handles the voice-based guessing."""
    guess = voice_guess()
    if guess is not None:
        entry.delete(0, tk.END)
        entry.insert(0, str(guess))
        check_guess()
    else:
        messagebox.showerror("Error", "Could not understand the voice input. Please try again.")

# ======== Step 8: Create GUI Window =========
root = tk.Tk()
root.title("🎤 AI Number Guessing Game")
root.geometry("500x400")

tk.Label(root, text="🔢 Guess a number between 1 and 100:", font=("Arial", 14)).pack(pady=10)
entry = tk.Entry(root, font=("Arial", 12))
entry.pack(pady=5)

check_button = tk.Button(root, text="✅ Check Guess", font=("Arial", 12), command=check_guess)
check_button.pack(pady=5)

voice_button = tk.Button(root, text="🎤 Guess with Voice", font=("Arial", 12), command=voice_guess_handler)
voice_button.pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
result_label.pack(pady=10)

root.mainloop()
