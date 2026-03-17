#!/usr/bin/env python
"""
Universal deployment inference script for cancer grade and TP53 mutation prediction.

Examples
--------
# Predict reusable components for grade pipeline
python predict.py --task grade --mode components --input expression.csv --output grade_components.csv

# Predict grade labels
python predict.py --task grade --mode predict --input expression.csv --output grade_predictions.csv

# Predict reusable components for TP53 pipeline
python predict.py --task tp53 --mode components --input expression.csv --output tp53_components.csv

# Predict TP53 mutation labels
python predict.py --task tp53 --mode predict --input expression.csv --output tp53_predictions.csv
"""

import argparse
import json
import os
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch


class UniversalCancerPredictor:
    def __init__(self, deployment_dir: str = "deployment", task: str = "grade"):
        self.deployment_dir = deployment_dir
        self.task = task.lower()

        if self.task not in {"grade", "tp53"}:
            raise ValueError("task must be one of: 'grade', 'tp53'")

        self.task_dir = self._resolve_task_dir()
        self._load_artifacts()

    def _resolve_task_dir(self) -> str:
        task_dirs = {
            "grade": os.path.join(self.deployment_dir, "Cancer Grade"),
            "tp53": os.path.join(self.deployment_dir, "TP53 Mutation"),
        }
        return task_dirs[self.task]

    def _find_existing_file(self, candidates):
        for candidate in candidates:
            path = os.path.join(self.task_dir, candidate)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Could not find any of these files in {self.task_dir}: {candidates}"
        )

    def _load_artifacts(self):
        scaler_path = self._find_existing_file([
            "cgrade_scaler.pkl",
            "mutation_cgrade_scaler.pkl",
            "scaler.pkl",
        ])
        phate_path = self._find_existing_file([
            "cgrade_phate.pkl",
            "mutation_cgrade_phate.pkl",
            "mutation_phate.pkl",
            "supervised_phate.pkl",
            "phate.pkl",
        ])
        component_predictor_path = self._find_existing_file([
            "cgrade_component_predictor_mlp.pt",
            "mutation_cgrade_component_predictor_mlp.pt",
            "component_predictor_mlp.pt",
            "feature_selector_mlp.pt",
        ])
        topk_path = self._find_existing_file([
            "cgrade_top_k_components.json",
            "mutation_cgrade_top_k_components.json",
            "top_k_components.json",
        ])
        gene_map_path = self._find_existing_file([
            "cgrade_component_gene_mapping.json",
            "mutation_cgrade_component_gene_mapping.json",
            "component_gene_mapping.json",
        ])
        config_path = self._find_existing_file([
            "cgrade_phate_supervision_config.json",
            "mutation_cgrade_phate_supervision_config.json",
            "phate_supervision_config.json",
        ])

        if self.task == "grade":
            predictor_path = self._find_existing_file([
                "cgrade_xgboost_grade_predictor.pkl",
                "xgboost_grade_predictor.pkl",
            ])
        else:
            predictor_path = self._find_existing_file([
                "mutation_cgrade_xgboost_grade_predictor.pkl",
                "tp53_predictor.pkl",
                "xgboost_tp53_predictor.pkl",
            ])

        self.scaler = joblib.load(scaler_path)
        self.phate_model = joblib.load(phate_path)

        try:
            self.component_predictor = torch.jit.load(component_predictor_path)
        except RuntimeError:
            self.component_predictor = torch.load(component_predictor_path, map_location="cpu")
            self.component_predictor.eval()

        self.predictor_model = joblib.load(predictor_path)

        with open(topk_path, "r") as f:
            self.top_k_indices = json.load(f)

        with open(gene_map_path, "r") as f:
            self.gene_mapping = json.load(f)

        with open(config_path, "r") as f:
            self.phate_config = json.load(f)

        print(f"✓ Loaded artifacts for task '{self.task}' from {self.task_dir}")

    def _prepare_supervision_labels(self, n_samples: int, y_labels: Optional[np.ndarray] = None) -> np.ndarray:
        if y_labels is None:
            y_onehot = np.zeros((n_samples, 2))
            y_onehot[:, 0] = 1.0
            return y_onehot

        y_labels = np.asarray(y_labels).astype(int).flatten()
        y_onehot = np.zeros((n_samples, 2))
        y_onehot[np.arange(n_samples), y_labels] = 1.0
        return y_onehot

    def predict_components(self, X_raw: np.ndarray, y_labels: Optional[np.ndarray] = None) -> Dict:
        n_samples = X_raw.shape[0]

        X_normalized = self.scaler.transform(X_raw)

        y_onehot = self._prepare_supervision_labels(n_samples, y_labels=y_labels)
        supervision_weight = self.phate_config.get("supervision_weight", 5.0)
        X_with_labels = np.concatenate([X_normalized, y_onehot * supervision_weight], axis=1)

        X_phate = self.phate_model.transform(X_with_labels)
        X_topk = X_phate[:, self.top_k_indices]

        X_topk_tensor = torch.FloatTensor(X_topk)
        with torch.no_grad():
            component_output = self.component_predictor(X_topk_tensor)

        if isinstance(component_output, torch.Tensor):
            predicted_components = component_output.cpu().numpy()
        else:
            predicted_components = np.asarray(component_output)

        interpretability = {}
        for i in range(min(3, n_samples)):
            sample_genes = {}
            for top_k_idx in self.top_k_indices[:5]:
                comp_key = f"component_{top_k_idx}"
                if comp_key in self.gene_mapping:
                    sample_genes[comp_key] = self.gene_mapping[comp_key].get("top_10_genes", [])[:3]
            interpretability[f"sample_{i}"] = sample_genes

        return {
            "predicted_components": predicted_components,
            "top_components_used": self.top_k_indices,
            "interpretability_sample": interpretability,
            "n_samples": n_samples,
        }

    def predict_task(self, X_raw: np.ndarray, y_labels: Optional[np.ndarray] = None) -> Dict:
        component_results = self.predict_components(X_raw, y_labels=y_labels)
        predicted_components = component_results["predicted_components"]

        predictions = self.predictor_model.predict(predicted_components)
        probabilities = self.predictor_model.predict_proba(predicted_components)[:, 1]

        if self.task == "grade":
            return {
                "grade_predictions": predictions,
                "grade_probabilities": probabilities,
                **component_results,
            }

        return {
            "tp53_predictions": predictions,
            "tp53_probabilities": probabilities,
            **component_results,
        }


def main():
    parser = argparse.ArgumentParser(description="Universal cancer prediction pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input CSV with raw expression (rows=samples, cols=genes)")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--deployment-dir", type=str, default="deployment", help="Root deployment directory")
    parser.add_argument("--task", type=str, required=True, choices=["grade", "tp53"], help="Prediction task")
    parser.add_argument("--mode", type=str, default="predict", choices=["predict", "components"], help="Run full prediction or output components only")
    parser.add_argument("--labels", type=str, default=None, help="Optional labels CSV for supervised PHATE transform")

    args = parser.parse_args()

    X_raw = pd.read_csv(args.input, index_col=0).values

    y_labels = None
    if args.labels:
        y_labels = pd.read_csv(args.labels, index_col=0).values.flatten()

    predictor = UniversalCancerPredictor(
        deployment_dir=args.deployment_dir,
        task=args.task,
    )

    if args.mode == "components":
        results = predictor.predict_components(X_raw, y_labels=y_labels)
        component_df = pd.DataFrame(
            results["predicted_components"],
            index=pd.read_csv(args.input, index_col=0).index,
            columns=[f"pred_component_{i}" for i in range(results["predicted_components"].shape[1])],
        )
        component_df.to_csv(args.output)
        print(f"Component embeddings saved to {args.output}")
        return

    results = predictor.predict_task(X_raw, y_labels=y_labels)

    if args.task == "grade":
        output_df = pd.DataFrame(
            {
                "grade_prediction": results["grade_predictions"],
                "high_risk_probability": results["grade_probabilities"],
            },
            index=pd.read_csv(args.input, index_col=0).index,
        )
    else:
        output_df = pd.DataFrame(
            {
                "tp53_prediction": results["tp53_predictions"],
                "tp53_probability": results["tp53_probabilities"],
            },
            index=pd.read_csv(args.input, index_col=0).index,
        )

    output_df.to_csv(args.output)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
