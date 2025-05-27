import subprocess
import webbrowser
import time
import sys

def run_streamlit_app():
    # Open Streamlit app subprocess
    process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'heart_disease_prediction_app.py'])

    # Wait a few seconds for the server to start
    time.sleep(3)

    # Open the app in the default web browser
    webbrowser.open_new("http://localhost:8501")

    # Wait for the Streamlit process to finish
    process.communicate()

if __name__ == '__main__':
    run_streamlit_app()
  