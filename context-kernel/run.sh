#!/bin/bash
# Run the FastAPI or Streamlit demo

if [ "$1" == "ui" ]; then
    streamlit run frontend/ui_app.py
else
    python app/main.py
fi
