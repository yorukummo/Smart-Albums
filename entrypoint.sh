#!/bin/bash
if [ "$MODE" = "train" ]; then
    python train.py --config config.json
else
    uvicorn api_service:app --host 0.0.0.0 --port 8000
fi
