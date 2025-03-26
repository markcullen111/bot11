#!/bin/bash

# Start MLflow tracking server in the background
mlflow ui --host 0.0.0.0 --port 5000 &

# Start the main trading bot application in the background
python app/main.py &

# Start the Streamlit UI (this will stay in the foreground)
streamlit run app/streamlit_app/app_ui.py --server.port 8501 --server.address 0.0.0.0

# Keep the container running
wait 