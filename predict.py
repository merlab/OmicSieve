#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost  # noqa: F401 - required when loading pickled XGBoost models


TASK_DEFAULT_METHOD = {
    "grade": "kpca_rbf",
    "tp53": "kpca_cosine",
}

TASK_DIR = {
    "grade": "deployment_grade",
    "tp53": "deployment_tp53",
}

TASK_MODEL_NAME = {
    "grade": "xgboost_grade_predictor_cv.pkl",
    "tp53": "xgboost_tp53_predictor_cv.pkl",
}

TASK_LABEL_NAME = {
    "grade": "grade_prediction",
    "tp53": "tp53_prediction",
}

TASK_PROBA_NAME = {
    "grade": "high_risk_probability",
    "tp53": "tp53_mutation_probability",
}


class FeatureAttentionBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.4, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(32, dim // 2)
        se_hidden = max(16, int(dim * se_ratio))
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.channel_attn = nn.Sequential(
            nn.Linear(dim, se_hidden),
            nn.ReLU(),
            nn.Linear(se_hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        x = x + (self.ff(h) * self.gate(h))
        scale = self.channel_attn(x)
        return x * scale + x


class AttentionFeatureSelectorMLP(nn.Module):
    def __init__(
        self,
        n_input_genes: int,
        n_output_components: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.4,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256, 128]

        self.stem = nn.Sequential(
            nn.Linear(n_input_genes, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        blocks = []
        transitions = []
        for i, dim in enumerate(hidden_dims):
            blocks.append(FeatureAttentionBlock(dim, dropout=dropout))
            if i < len(hidden_dims) - 1:
                transitions.append(
                    nn.Sequential(
                        nn.Linear(dim, hidden_dims[i + 1]),
                        nn.BatchNorm1d(hidden_dims[i + 1]),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.transitions = nn.ModuleList(transitions)
        self.component_head = nn.Linear(hidden_dims[-1], n_output_components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i < len(self.transitions):
                h = self.transitions[i](h)
        return self.component_head(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OmicSieve inference on an omics feature CSV.")
    parser.add_argument("--task", choices=["grade", "tp53"], required=True)
    parser.add_argument(
        "--mode",
        choices=["components", "predict"],
        required=True,
        help="'components' only needs the encoder package; 'predict' also needs a downstream classifier.",
    )
    parser.add_argument("--input", required=True, help="CSV with samples as rows and omics features as columns.")
    parser.add_argument("--output", required=True, help="Output path. Use .csv or .npz.")
    parser.add_argument(
        "--model-root",
        default=None,
        help="Artifact folder. Defaults to deployment_grade or deployment_tp53 based on --task.",
    )
    parser.add_argument(
        "--method",
        choices=["kpca_rbf", "kpca_cosine"],
        default=None,
        help="Component method. Defaults to grade=kpca_rbf and tp53=kpca_cosine.",
    )
    parser.add_argument(
        "--gene-order",
        default=None,
        help="JSON/TXT/CSV file containing the training gene order. Defaults to <model-root>/gene_order.json.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for the MLP component predictor.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def method_display_name(method: str) -> str:
    return {"kpca_rbf": "KPCA_RBF", "kpca_cosine": "KPCA_COSINE"}[method]


def normalize_gene_name(name: object) -> str:
    return str(name).strip().strip('"')


def load_gene_order(path: Path) -> list[str]:
    require_file(path, "gene-order file")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            for key in ("genes", "gene_order", "gene_names"):
                if key in data:
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"Gene-order JSON must be a list or contain a genes/gene_order/gene_names list: {path}")
        return [normalize_gene_name(g) for g in data]

    if path.suffix.lower() in {".txt", ".tsv"}:
        return [normalize_gene_name(line) for line in path.read_text().splitlines() if line.strip()]

    frame = pd.read_csv(path)
    if frame.shape[1] == 0:
        raise ValueError(f"Gene-order CSV has no columns: {path}")
    return [normalize_gene_name(g) for g in frame.iloc[:, 0].tolist()]


def read_expression_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    if frame.empty:
        raise ValueError(f"Input CSV is empty: {path}")
    frame.columns = [normalize_gene_name(c) for c in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].unique().tolist()
        raise ValueError(f"Input CSV contains duplicated gene columns, e.g. {duplicates[:5]}")
    return frame


def align_gene_order(frame: pd.DataFrame, gene_order: list[str] | None, n_expected: int) -> pd.DataFrame:
    if gene_order is None:
        if frame.shape[1] != n_expected:
            raise ValueError(
                "No gene-order file was found, and the input column count does not match the model. "
                f"Expected {n_expected} genes, got {frame.shape[1]}. Provide --gene-order."
            )
        print(
            "Warning: no gene-order file found; using input columns as-is. "
            "Provide --gene-order to reorder by gene name.",
            file=sys.stderr,
        )
        return frame

    if len(gene_order) != n_expected:
        raise ValueError(f"Gene-order length is {len(gene_order)}, but the model expects {n_expected} genes.")
    if len(set(gene_order)) != len(gene_order):
        raise ValueError("Gene-order file contains duplicated genes.")

    missing = [gene for gene in gene_order if gene not in frame.columns]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Input CSV is missing {len(missing)} required genes. First missing genes: {preview}")

    expected_genes = set(gene_order)
    extra = [gene for gene in frame.columns if gene not in expected_genes]
    if extra:
        print(f"Warning: ignoring {len(extra)} extra input gene columns.", file=sys.stderr)

    return frame.loc[:, gene_order]


def infer_model_shape(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    input_weight = state_dict["stem.0.weight"]
    output_weight = state_dict["component_head.weight"]
    return int(input_weight.shape[1]), int(output_weight.shape[0])


def load_mlp_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], int, int, list[int]]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        n_input_genes = int(checkpoint.get("n_input_genes", infer_model_shape(state_dict)[0]))
        n_output_components = int(checkpoint.get("n_output_components", infer_model_shape(state_dict)[1]))
        hidden_dims = list(checkpoint.get("hidden_dims", [1024, 512, 256, 128]))
        return state_dict, n_input_genes, n_output_components, hidden_dims

    state_dict = checkpoint
    n_input_genes, n_output_components = infer_model_shape(state_dict)
    return state_dict, n_input_genes, n_output_components, [1024, 512, 256, 128]


def predict_components(
    model: AttentionFeatureSelectorMLP,
    values: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start : start + batch_size], dtype=torch.float32)
            chunks.append(model(batch).cpu().numpy())
    return np.vstack(chunks)


def build_component_frame(index: Iterable[object], components: np.ndarray) -> pd.DataFrame:
    columns = [f"pred_component_{i}" for i in range(components.shape[1])]
    return pd.DataFrame(components, index=index, columns=columns)


def write_output(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".npz":
        np.savez_compressed(
            output_path,
            values=frame.to_numpy(),
            sample_ids=frame.index.astype(str).to_numpy(),
            columns=np.array(frame.columns.astype(str).tolist()),
        )
        return
    frame.to_csv(output_path)


def main() -> None:
    args = parse_args()
    task = args.task
    method = args.method or TASK_DEFAULT_METHOD[task]
    model_root = Path(args.model_root or TASK_DIR[task])
    method_dir = model_root / method
    classifier_dir = model_root / method_display_name(method)

    scaler_path = require_file(model_root / "scaler.pkl", "scaler")
    mlp_path = require_file(method_dir / "component_predictor_attention_mlp.pt", "component predictor")

    state_dict, n_input_genes, n_output_components, hidden_dims = load_mlp_checkpoint(mlp_path)

    gene_order_path = Path(args.gene_order) if args.gene_order else model_root / "gene_order.json"
    gene_order = load_gene_order(gene_order_path) if gene_order_path.exists() else None

    expression = read_expression_csv(Path(args.input))
    expression = align_gene_order(expression, gene_order, n_input_genes)

    scaler = joblib.load(scaler_path)
    x_scaled = scaler.transform(expression.to_numpy(dtype=np.float32))

    mlp = AttentionFeatureSelectorMLP(
        n_input_genes=n_input_genes,
        n_output_components=n_output_components,
        hidden_dims=hidden_dims,
    )
    mlp.load_state_dict(state_dict)
    components = predict_components(mlp, x_scaled, args.batch_size)

    if args.mode == "components":
        output = build_component_frame(expression.index, components)
    else:
        classifier_path = require_file(classifier_dir / TASK_MODEL_NAME[task], "XGBoost classifier")
        classifier = joblib.load(classifier_path)
        labels = classifier.predict(components).astype(int)
        probabilities = classifier.predict_proba(components)[:, 1]
        output = pd.DataFrame(
            {
                TASK_LABEL_NAME[task]: labels,
                TASK_PROBA_NAME[task]: probabilities,
            },
            index=expression.index,
        )

    output.index.name = expression.index.name or "sample_id"
    output_path = Path(args.output)
    write_output(output, output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
