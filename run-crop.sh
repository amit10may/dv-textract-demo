#!/bin/bash

# Option 1: Using nohup
nohup streamlit run app-crop.py --server.port 9501 --server.address 0.0.0.0 > streamlit.log 2>&1 &

# Print the process ID
echo $! > .pid
echo "Streamlit app is running with PID: $(cat .pid)"