#!/bin/bash
# Launch MLflow UI for Edge server

if [ "$1" == "edge" ] || [ -z "$1" ]; then
    echo "Starting MLflow UI for Edge server..."
    mlflow ui --backend-store-uri sqlite:///mlflow_edge.db --port 5000 --host 0.0.0.0
else
    echo "Usage: ./mlflow_ui.sh [edge]"
    echo "  edge   - Start MLflow UI for Edge server on port 5000"
fi
