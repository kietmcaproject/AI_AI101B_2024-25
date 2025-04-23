import tkinter as tk
from tkinter import scrolledtext
import requests
import json

# Function to send message to Rasa server and get response
def get_bot_response(message):
    url = "http://localhost:5005/webhooks/rest/webhook"
    payload = {
        "sender": "user",  # or a unique user id
        "message": message
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        messages = response.json()
        bot_response = "\n".join([msg.get("text", "") for msg in messages])
        return bot_response if bot_response else "🤖 Sorry, I didn't understand that."
    except Exception as e:
        return f"Error: {str(e)}"

# Function to handle sending messages in GUI
def send_message(event=None):
    user_msg = user_input.get()
    if user_msg.strip() == "":
        return
    chat_window.insert(tk.END, f"You: {user_msg}\n")
    response = get_bot_response(user_msg)
    chat_window.insert(tk.END, f"🤖 ChatBot: {response}\n\n")
    user_input.delete(0, tk.END)
    chat_window.see(tk.END)

# Create GUI
root = tk.Tk()
root.title("Rasa ChatBot GUI")
root.geometry("500x500")

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 14))
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_window.insert(tk.END, "🤖 ChatBot: Hello! Ask me anything.\n\n")

user_input = tk.Entry(root, font=("Arial", 14))
user_input.pack(padx=10, pady=5, fill=tk.X)
user_input.bind("<Return>", send_message)

send_btn = tk.Button(root, text="Send", font=("Arial", 12), command=send_message)
send_btn.pack(padx=10, pady=5)

root.mainloop()