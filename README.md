# Cancer Multi-Task Prediction Pipeline - Deployment Guide

This deployment package provides a complete cancer prediction pipeline with three capabilities:
1. **Component Prediction**: Extract reusable 30-dimensional embeddings from gene expression
2. **Grade Prediction**: Predict cancer histological grade (low vs high risk)
3. **TP53 Mutation Prediction**: Predict TP53 mutation status

## Package Contents

### Required Files
1. **Grade Prediction**:
    - `cgrade_scaler.pkl` - StandardScaler for gene expression normalization
    - `cgrade_component_predictor_mlp.pt` - PyTorch MLP model for component prediction
    - `cgrade_top_k_components.json` - Top-30 PHATE component indices
    - `cgrade_xgboost_grade_predictor.pkl` - XGBoost model for grade classification
    - `cgrade_component_gene_mapping.json` - Maps components to top contributing genes
    - `cgrade_phate.pkl` - PHATE dimensionality reduction model
    - `cgrade_phate_supervision_config.json` - PHATE configuration parameters

2. **TP53 Mutatuion Prediction**:
    - `mutation_scaler.pkl`
    - `mutation_component_predictor_mlp.pt`
    - `mutation_top_k_components.json`
    - `mutation_xgboost_grade_predictor.pkl`
    - `mutation_component_gene_mapping.json`
    - `mutation_phate.pkl`
    - `mutation_phate_supervision_config.json`

### Download Pretrained Models

Use `gdown` to download the required deployment files:

```bash
gdown 1VoiOg9sqwVf0HbVefb-_YN7_tCE6mO7-  # cgrade_phate.pkl
gdown 14s8pPSfHmcheKTTStfW2Lqy8F7zGw9p6  # cgrade_component_predictor_mlp.pt 
gdown 12KgrbBWlVxycZRYwH_BTvIaEaKPeqEoM  # mutation_phate.pkl
gdown 11cCVYPjG783Rug5ucwlqzZT_8HDNsxhf  # mutation_component_predictor_mlp.pt 

gdown 1Mkgy--OAP3hq_ke42mBCD_QFWAfa_mLT  # mutation_component_predictor_mlp_k50.pt
gdown 1zf2WAx5q0gieIRJBIGy2Gye8dGUJIvV0  # mutation_phate_k50.pkl

```

After downloading, place the files in the following directories:

```
deployment/
├────Cancer Grade
        ├── cgrade_phate.pkl
        ├── cgrade_component_predictor_mlp.pt
├────TP53 Mutation
        ├── mutation_phate.pkl
        ├── mutation_component_predictor_mlp.pt
```

### Generated Files
- `predict.py` - Standalone universal inference script


---

### Installation

```bash
# Install required packages
pip install numpy pandas scikit-learn xgboost torch joblib
```

### Basic Usage

```bash
# Predict components (reusable embeddings) for grade pipeline
python predict.py --task grade --mode components --input your_data.csv --output grade_components.csv

# Predict cancer grade
python predict.py --task grade --mode predict --input your_data.csv --output grade_predictions.csv

# Predict TP53 mutation status
python predict.py --task tp53 --mode predict --input your_data.csv --output tp53_predictions.csv
```



---

## Input Data Format

Your input CSV file should have:
- **Rows**: Samples/patients
- **Columns**: 19,310 gene expression values (same order as training data)
- **Index**: Sample IDs (first column)

Example structure:
```csv
sample_id,gene1,gene2,gene3,...,gene19310
TCGA-A1-A0SB-01,5.234,3.142,7.891,...,2.456
TCGA-A1-A0SD-01,4.567,2.987,6.543,...,3.789
```

**Important**: Gene expression values should be **raw counts** or **normalized** in the same way as the training data.

---

##  Detailed Usage

### 1. Component Prediction (Embeddings)

Extract 30-dimensional component embeddings for downstream tasks.

Grade pipeline:

```bash
python predict.py \
    --task grade \
    --mode components \
    --input expression_data.csv \
    --output grade_components.csv
```

TP53 pipeline:

```bash
python predict.py \
    --task tp53 \
    --mode components \
    --input expression_data.csv \
    --output tp53_components.csv
```

**Output**: CSV with 30 columns (pred_component_0 to pred_component_29)

**Use cases**:
- Transfer learning to new tasks
- Visualization and clustering
- Custom downstream predictors
- Biomarker discovery

---

### 2. Grade Prediction

Predict cancer histological grade (binary classification):

```bash
python predict.py \
    --task grade \
    --mode predict \
    --input expression_data.csv \
    --output grade_predictions.csv
```

**Output**: CSV with two columns:
- `grade_prediction`: 0 (low risk) or 1 (high risk)
- `high_risk_probability`: Probability of high-risk grade (0-1)

**Performance Metrics**:
- Test Accuracy: 83.83%
- Test F1 Score: 85.25%
- Test Balanced Accuracy: 73.12%

---

### 3. TP53 Mutation Prediction

Predict TP53 mutation status:

```bash
python predict.py \
    --task tp53 \
    --mode predict \
    --input expression_data.csv \
    --output tp53_predictions.csv
```

**Output**: CSV with two columns:
- `tp53_prediction`: 0 (wild-type) or 1 (mutated)
- `tp53_probability`: Probability of mutation (0-1)

**Performance Metrics**:
- Test Accuracy: 0.7327
- Test Balanced Accuracy: 0.7203
- Test F1 (weighted): 0.7341

---

## Python API Usage

For programmatic access, use the `UniversalCancerPredictor` class:

```python
from predict import UniversalCancerPredictor
import pandas as pd

# Load your data (samples × genes)
X_raw = pd.read_csv('expression_data.csv', index_col=0).values

# Grade pipeline
grade_pipe = UniversalCancerPredictor(deployment_dir='deployment', task='grade')
grade_components = grade_pipe.predict_components(X_raw)
grade_results = grade_pipe.predict_task(X_raw)

print(grade_results['grade_predictions'])
print(grade_results['grade_probabilities'])

# TP53 pipeline
tp53_pipe = UniversalCancerPredictor(deployment_dir='deployment', task='tp53')
tp53_components = tp53_pipe.predict_components(X_raw)
tp53_results = tp53_pipe.predict_task(X_raw)

print(tp53_results['tp53_predictions'])
print(tp53_results['tp53_probabilities'])
```

---

## Gene Interpretation

Each component is associated with its top contributing genes. To find which genes drive a particular component:

```python
import json

# Example path for one task-specific deployment folder
with open('deployment/Cancer Grade/cgrade_component_gene_mapping.json', 'r') as f:
    gene_map = json.load(f)

# Get top genes for component 42
component_key = 'component_42'
top_genes = gene_map[component_key]['top_10_genes']

for gene_info in top_genes:
    print(f"{gene_info['gene_name']}: {gene_info['abs_coefficient']:.4f}")
```

Or use the built-in method by inspecting the returned interpretability fields from the predictor output.

---

## Model Performance

### Component Predictor
- **Test MSE**: ~0.0000 (excellent reconstruction of PHATE components)

### Grade Predictor (XGBoost on Components)
- **Test Accuracy**: 87.47%
- **Test F1 (weighted) Score**: 87.74%
- **Test Balanced Accuracy**: 73.70%

### TP53 Mutation Predictor
**30 component:**
- **Test Accuracy**: 73.27%
- **Test Balanced Accuracy**: 72.03%
- **Test F1 (weighted)**: 73,41%

**50 component:**
- **Test Accuracy**: 0.7493
- **Test Balanced Accuracy**: 0.7419
- **Test F1 (weighted)**: 0.7514


### Execution time details

**Grade Prediction**
- **PHATE 100 Component Extraction:** 226.42s / 3.7m
- **SHAP computation time:** 3.16s
- **MLP Component Training:** 242.71s / 4.04m
-  **MLP infer time/sample:** 0.147ms
-  **XGBoost infer time/sample:** 0.027ms

**TP53 Mutation Prediction**
- **PHATE 100 Component Extraction:** 288.73s / 4.8m
- **SHAP computation time:** 8.67s
- **MLP Component Training:**  1772.09s / 12.8m
-  **MLP infer time/sample:** 0.169ms
-  **XGBoost infer time/sample:** 0.031ms
---

<!-- ##  Citation

If you use this pipeline in your research, please cite:

```
[Add appropriate citation here]
``` -->

## Pipeline Architecture
![Architecture](Architecture.jpg "Architecture Pipeline")


The pipeline uses a two-stage approach:
1. **Stage 1**: MLP predicts 30 supervised PHATE components from raw genes
2. **Stage 2**: Downstream XGBoost classifiers use these components for predictionsMLP Component Predictor

This design allows you to:
- Reuse components for new prediction tasks
- Interpret results via gene mapping
- Achieve strong performance with compact representations

---
