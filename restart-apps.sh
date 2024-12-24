#!/bin/bash

# Kill existing Streamlit processes
echo "Stopping existing Streamlit applications..."
pkill -f "streamlit run app.py"
pkill -f "streamlit run app-crop.py"

# Wait a moment to ensure processes are stopped
sleep 2

# Start both apps using their respective run scripts
echo "Starting app.py..."
./run.sh

echo "Starting app-crop.py..."
./run-crop.sh

# Wait a moment for apps to start
sleep 3

# Check if apps are running
if pgrep -f "streamlit run app.py" > /dev/null
then
    echo "app.py is running on http://0.0.0.0:8501"
else
    echo "Error: app.py failed to start"
fi

if pgrep -f "streamlit run app-crop.py" > /dev/null
then
    echo "app-crop.py is running on http://0.0.0.0:9501"
else
    echo "Error: app-crop.py failed to start"
fi