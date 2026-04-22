# OmicSieve

OmicSieve is a generalized unsupervised representation-learning pipeline for omics data. It uses non-linear compression to convert high-dimensional molecular profiles into compact embeddings that can be reused for many downstream tasks.

The saved OmicSieve MLP model lets users project their own samples into the same compressed embedding space. Those embeddings can then be used as input features for any downstream analysis or model, such as binary or multi-class classification, regression, clustering, survival analysis, visualization, or biomarker discovery.

Cancer grade and TP53 mutation prediction from RNA-seq gene expression are examples of downstream analyses that can be built on top of the learned embeddings. OmicSieve itself is not limited to binary classification.

![OmicSieve architecture](Architecture.jpg "OmicSieve architecture")

## What OmicSieve Provides

- A saved non-linear MLP encoder for predicting compressed omics embeddings.
- A fixed feature-order file so input features can be reordered correctly.
- Component–gene association files for post hoc interpretation of compressed embeddings.

## Package Contents

Only a few files are required to generate compressed embeddings from new samples. After downloading the encoder weights, the expected layout is:

```text
deployment_grade/
|-- gene_order.json
|-- scaler.pkl
|-- kpca_rbf/
|   `-- component_predictor_attention_mlp.pt
`-- kpca_cosine/
    `-- component_predictor_attention_mlp.pt

deployment_tp53/
|-- gene_order.json
|-- scaler.pkl
|-- kpca_rbf/
|   `-- component_predictor_attention_mlp.pt
`-- kpca_cosine/
    `-- component_predictor_attention_mlp.pt
```

`gene_order.json` aligns incoming features, `scaler.pkl` applies the training normalization, and `component_predictor_attention_mlp.pt` generates the 50-dimensional compressed embedding. Configuration files, ranking plots, and training curves are useful for analysis and reproducibility, but are not needed by `predict.py` to embed new samples.

Component–gene association files can be provided separately for grade and TP53 interpretation.

## Component–Gene Association Analysis

To support biological interpretation, OmicSieve provides correlation-based component–gene association analyses that rank genes by their association with each retained latent component. For each selected component, genes are ranked by absolute Spearman correlation across training samples and the top associated genes are saved in task-specific files such as:

```text
component_gene_association_kpca_rbf.json
component_gene_association_kpca_cosine.json
```

These associations are intended as interpretability aids rather than exact mechanistic decompositions of the latent space.

## Download Large Files

```bash
mkdir -p deployment_grade/kpca_rbf deployment_grade/kpca_cosine
mkdir -p deployment_tp53/kpca_rbf deployment_tp53/kpca_cosine

# Grade encoder - RBF
gdown 1Ap7CXaGOjPunefzySrFsuVH4jF6Q_BHS -O deployment_grade/kpca_rbf/component_predictor_attention_mlp.pt
# Grade KPCA - RBF
gdown 1JJoa-3b3J1rgyHNcXNj3M1yojE7I3hkE -O deployment_grade/kpca_rbf/kpca.pkl
# Grade encoder - Cosine
gdown 1GLLGWkEVakDsrQtvrwnSwJEVHejCHC-W -O deployment_grade/kpca_cosine/component_predictor_attention_mlp.pt
# Grade KPCA - Cosine
gdown 1wi3hnXcX3CycUcv4eSV58vZyBuCdsdpR -O deployment_grade/kpca_cosine/kpca.pkl

# TP53 encoder - RBF
gdown 1I0HNoTgDUVpG5bwAp8GX1_g3NoukLXVA -O deployment_tp53/kpca_rbf/component_predictor_attention_mlp.pt
# TP53 KPCA - RBF
gdown 1TKa-YrBWl0UUK_CORors3q7kRAjyEkXK -O deployment_tp53/kpca_rbf/kpca.pkl
# TP53 encoder - Cosine
gdown 145GibC68P-g-Yumf_L1h-cyrl90dhGIH -O deployment_tp53/kpca_cosine/component_predictor_attention_mlp.pt
# TP53 KPCA - Cosine
gdown 1-fRfVsm8GU9kW1Q5Jrgxqi58TACaBurP -O deployment_tp53/kpca_cosine/kpca.pkl
```

## Input Format

Input CSV files should have samples as rows and omics features as columns. For the provided examples, these features are RNA-seq genes. The first column is treated as the sample ID.

```csv
sample_id,TP53,BRCA1,BRCA2,...
TCGA-02-0047-01,4.31,2.18,3.92,...
TCGA-02-0055-01,5.02,2.44,4.10,...
```

`predict.py` reorders input columns to the provided training feature order. The RNA-seq example model expects 19,310 genes.

## Install

```bash
pip install numpy pandas scikit-learn torch joblib
```

## Usage

Predict reusable compressed embeddings:

```bash
python predict.py --task grade --mode components --input your_data.csv --output grade_components.csv
```

Embeddings can also be saved as compressed NumPy archives:

```bash
python predict.py --task grade --mode components --input your_data.csv --output grade_components.npz
```

The output embeddings can be used for any downstream task. For example, you can train your own classifier or regressor on `grade_components.csv`:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

X = pd.read_csv("grade_components.csv", index_col=0)
y = pd.read_csv("your_labels.csv", index_col=0)["label"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)
```

Useful options:

```bash
--method kpca_rbf      # or kpca_cosine
--model-root PATH      # model artifact folder
--gene-order PATH      # JSON/TXT/CSV training feature order
```

Default methods:

- Grade: `kpca_rbf`
- TP53: `kpca_cosine`

## Output

Component mode writes 50 compressed embedding columns:

```text
pred_component_0 ... pred_component_49
```

## Notes

- The RNA-seq example model expects 19,310 gene-expression features.
- Use the provided `gene_order.json` to align input features before inference.
- Extra input columns are ignored when `gene_order.json` is provided.
- Missing required features stop inference with an error.
- For new downstream tasks, use `--mode components` and train your task-specific model on the generated embeddings.
