#!/bin/bash
# Installation and setup instructions

# 1. Install dependencies
pip install -r requirements.txt

# 2. Navigate to deployment directory
cd deployment

# 3. Make predictions on new data
# Option A: Python API
python3 << 'PYTHON_EOF'
from predict_grade import CancerGradePredictor
import numpy as np

# Example: Load your expression matrix
# X_new = np.loadtxt('your_expression_data.csv', delimiter=',', skiprows=1)

predictor = CancerGradePredictor(deployment_dir='.')
# results = predictor.predict_grade(X_new, y_labels=None)
print("✓ Predictor initialized successfully")
print("✓ Ready to make predictions on new data")
PYTHON_EOF

# Option B: Command line
# python3 predict_grade.py --input your_expression.csv --output predictions.csv

# 4. View metadata
cat metadata.json

# 5. Clean logs
rm -f *.log
