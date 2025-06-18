#!/bin/bash

# Optional: Create virtual environment (if needed)
# conda create -n deepfake-detector python=3.10 -y
# conda activate deepfake-detector

echo "Installing requirements..."
pip install -r requirements.txt

echo "Running training..."
python train_lora.py

echo "Running evaluation..."
python eval_lora.py

echo "Generating per-video scores (base & LoRA)..."
python generate_all_scores.py

echo "Done. All results and scores are generated in the results/ directory."
