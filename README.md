# Cancer Multi-Task Prediction Pipeline - Deployment Guide

This deployment package provides a complete cancer prediction pipeline with three capabilities:
1. **Component Prediction**: Extract reusable 30-dimensional embeddings from gene expression
2. **Grade Prediction**: Predict cancer histological grade (low vs high risk)
3. **TP53 Mutation Prediction**: Predict TP53 mutation status

## Package Contents

### Required Files
- `scaler.pkl` - StandardScaler for gene expression normalization
- `component_predictor_mlp.pt` - PyTorch MLP model for component prediction
- `top_k_components.json` - Top-30 PHATE component indices
- `xgboost_grade_predictor.pkl` - XGBoost model for grade classification
- `component_gene_mapping.json` - Maps components to top contributing genes
- `supervised_phate.pkl` - PHATE dimensionality reduction model
- `phate_supervision_config.json` - PHATE configuration parameters


### Download Pretrained Models

Use `gdown` to download the required deployment files:

```bash
gdown 1NYpS6_PlrFtUNXDOB-q0669zBVRnh7EU   # supervised_phate.pkl
gdown 1M61lFXz-kvpTh5RxrKD9oPIVet0BjLDV   # component_predictor_mlp.pt
gdown 16Fs7lzs5F6I2nMy5NHv9Zw6l5BlrBi0p   # feature_selector_mlp.pt
```

After downloading, place the files in the `deployment/` directory:

```
deployment/
├── supervised_phate.pkl
├── component_predictor_mlp.pt
└── feature_selector_mlp.pt
```

`
### Optional Files
- `tp53_predictor.pkl` - XGBoost model for TP53 mutation prediction


### Generated Files
- `metadata.json` - Pipeline metadata and performance metrics
- `predict_grade.py` - Standalone inference script

---


### Installation

```bash
# Install required packages
pip install numpy pandas scikit-learn xgboost torch joblib
```

### Basic Usage

```bash
# Predict components (reusable embeddings)
python predict_grade.py --input your_data.csv --output components.csv --mode components

# Predict cancer grade
python predict_grade.py --input your_data.csv --output grade_predictions.csv --mode grade

# Predict TP53 mutation status
python predict_grade.py --input your_data.csv --output tp53_predictions.csv --mode tp53
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

Extract 30-dimensional component embeddings for downstream tasks:

```bash
python predict_grade.py \
    --input expression_data.csv \
    --output components.csv \
    --mode components
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
python predict_grade.py \
    --input expression_data.csv \
    --output grade_predictions.csv \
    --mode grade
```

**Output**: CSV with two columns:
- `grade_prediction`: 0 (low risk) or 1 (high risk)
- `high_risk_probability`: Probability of high-risk grade (0-1)

**Performance Metrics** (from metadata.json):
- Test Accuracy: ~89%
- Test F1 Score: ~87%

---

### 3. TP53 Mutation Prediction

Predict TP53 mutation status:

```bash
python predict_grade.py \
    --input expression_data.csv \
    --output tp53_predictions.csv \
    --mode tp53
```

**Output**: CSV with two columns:
- `tp53_prediction`: 0 (wild-type) or 1 (mutated)
- `tp53_probability`: Probability of mutation (0-1)

**Performance Metrics** (from metadata.json):
- Test Accuracy: ~60%
- Test F1 Score: ~57%

---

## Python API Usage

For programmatic access, use the `CancerFeaturePipeline` class:

```python
from predict_grade import CancerFeaturePipeline
import pandas as pd

# Load the pipeline
pipe = CancerFeaturePipeline(deployment_dir='deployment')

# Load your data (samples × genes)
X_raw = pd.read_csv('expression_data.csv', index_col=0).values

# 1. Get component embeddings
components = pipe.predict_components(X_raw)
print(f"Component shape: {components.shape}")  # (n_samples, 30)

# 2. Predict cancer grade
grade_results = pipe.predict_grade(X_raw)
print(f"Predictions: {grade_results['grade_predictions']}")
print(f"Probabilities: {grade_results['grade_probabilities']}")

# 3. Predict TP53 mutation
tp53_results = pipe.predict_tp53(X_raw)
print(f"TP53 predictions: {tp53_results['tp53_predictions']}")
print(f"TP53 probabilities: {tp53_results['tp53_probabilities']}")

# 4. Map components to genes
gene_mapping = pipe.map_components_to_genes(
    components, 
    top_n_components=10,  # Top 10 strongest components
    top_n_genes=10        # Top 10 genes per component
)
print(gene_mapping['sample_0'])  # Gene contributors for first sample
```

---


## Gene Interpretation

Each component is associated with its top contributing genes. To find which genes drive a particular component:

```python
import json

# Load gene mapping
with open('deployment/component_gene_mapping.json', 'r') as f:
    gene_map = json.load(f)

# Get top genes for component 42
component_key = 'component_42'
top_genes = gene_map[component_key]['top_10_genes']

for gene_info in top_genes:
    print(f"{gene_info['gene_name']}: {gene_info['abs_coefficient']:.4f}")
```

Or use the built-in method:

```python
# Get gene mapping for specific samples
components = pipe.predict_components(X_raw)
gene_mapping = pipe.map_components_to_genes(
    components, 
    top_n_components=5,   # Top 5 components per sample
    top_n_genes=10        # Top 10 genes per component
)

# Examine first sample
sample_0_genes = gene_mapping['sample_0']
for comp in sample_0_genes:
    print(f"\nComponent {comp['component_index']} (score: {comp['component_score']:.3f})")
    for gene in comp['top_genes'][:5]:
        print(f"  - {gene['gene_name']}: {gene['abs_coefficient']:.4f}")
```


---

## Model Performance

### Component Predictor
- **Test MSE**: ~0.0000 (excellent reconstruction of PHATE components)

### Grade Predictor (XGBoost on Components)
- **Test Accuracy**: 83.83%
- **Test F1 Score**: 85.25%
- *est Balanced Accuracy**: 73.12%
- **Confusion Matrix**: Low false positive/negative rates

### TP53 Mutation Predictor (SVM on Components)
- **Test Accuracy**: 63.34%
- **Test F1 Score**: 65.66%
- **Test AUC-ROC**: 71.29%
- **Note**: TP53 mutation is a challenging task due to class imbalance

Performance metrics are stored in `metadata.json`.

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
2. **Stage 2**: Downstream XGBoost classifiers use these components for predictions

This design allows you to:
- Reuse components for new prediction tasks
- Interpret results via gene mapping
- Achieve strong performance with compact representations

---
