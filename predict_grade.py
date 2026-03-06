#!/usr/bin/env python
"""
Deployment inference script for reusable component prediction and downstream tasks.
Usage examples:
  python predict_grade.py --input expression.csv --output components.csv --mode components
  python predict_grade.py --input expression.csv --output grade.csv --mode grade
  python predict_grade.py --input expression.csv --output tp53.csv --mode tp53
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib


class FeatureSelectorMLP(nn.Module):
    def __init__(self, n_input_genes, n_output_components, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        layers = []
        prev_dim = n_input_genes
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_output_components))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CancerFeaturePipeline:
    def __init__(self, deployment_dir='deployment'):
        self.deployment_dir = deployment_dir
        self._load_artifacts()

    def _load_artifacts(self):
        self.scaler = joblib.load(os.path.join(self.deployment_dir, 'scaler.pkl'))
        self.grade_model = joblib.load(os.path.join(self.deployment_dir, 'xgboost_grade_predictor.pkl'))

        with open(os.path.join(self.deployment_dir, 'top_k_components.json'), 'r') as f:
            top_cfg = json.load(f)
        self.top_k_indices = top_cfg.get('component_indices', top_cfg)

        with open(os.path.join(self.deployment_dir, 'component_gene_mapping.json'), 'r') as f:
            self.gene_mapping = json.load(f)

        checkpoint = torch.load(
            os.path.join(self.deployment_dir, 'component_predictor_mlp.pt'),
            map_location='cpu'
        )

        self.component_predictor = FeatureSelectorMLP(
            n_input_genes=checkpoint['n_input_genes'],
            n_output_components=checkpoint['n_output_components'],
            hidden_dims=checkpoint.get('hidden_dims', [1024, 512, 256]),
        )
        self.component_predictor.load_state_dict(checkpoint['model_state_dict'])
        self.component_predictor.eval()

        mutation_model_path = os.path.join(self.deployment_dir, 'mutation_predictor_xgb.pkl')
        tp53_model_path = os.path.join(self.deployment_dir, 'tp53_predictor.pkl')
        self.has_tp53_model = os.path.exists(tp53_model_path)

        if self.has_tp53_model:
            self.tp53_model = joblib.load(tp53_model_path)
        else:
            self.tp53_model = None

        print(f"✓ Loaded artifacts from {self.deployment_dir}/")

    def predict_components(self, X_raw):
        X_scaled = self.scaler.transform(X_raw)
        with torch.no_grad():
            pred_components = self.component_predictor(torch.FloatTensor(X_scaled)).numpy()
        return pred_components

    def map_components_to_genes(self, component_scores, top_n_components=10, top_n_genes=10):
        output = {}
        for i in range(component_scores.shape[0]):
            strongest = np.argsort(-np.abs(component_scores[i]))[:top_n_components]
            sample_report = []
            for local_idx in strongest:
                comp_global_idx = self.top_k_indices[local_idx]
                comp_key = f'component_{comp_global_idx}'
                gene_info = self.gene_mapping.get(comp_key, {}).get('top_10_genes', [])[:top_n_genes]
                sample_report.append({
                    'component_index': int(comp_global_idx),
                    'component_score': float(component_scores[i, local_idx]),
                    'top_genes': gene_info,
                })
            output[f'sample_{i}'] = sample_report
        return output

    def predict_grade(self, X_raw):
        pred_components = self.predict_components(X_raw)
        grade_predictions = self.grade_model.predict(pred_components)
        grade_probabilities = self.grade_model.predict_proba(pred_components)[:, 1]

        return {
            'grade_predictions': grade_predictions,
            'grade_probabilities': grade_probabilities,
            'predicted_components': pred_components,
        }

    def predict_mutation(self, X_raw):
        if not self.has_mutation_model:
            raise RuntimeError('Mutation model artifacts are not available in deployment directory.')
    def predict_mutation(self, X_raw):
    def predict_tp53(self, X_raw):
        if not self.has_tp53_model:
            raise RuntimeError('TP53 model artifact is not available in deployment directory.')
        return {
            'mutation_predictions': mut_pred,
        tp53_pred = self.tp53_model.predict(pred_components)
        tp53_proba = self.tp53_model.predict_proba(pred_components)[:, 1]

        return {
            'tp53_predictions': tp53_pred,
            'tp53_probabilities': tp53_proba,
            'predicted_components': pred_components,
        }
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')

def main():
    parser = argparse.ArgumentParser(description='Cancer multi-task feature pipeline')
    parser.add_argument('--input', type=str, required=True, help='Input CSV with raw expression (rows=samples, cols=genes)')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--deployment-dir', type=str, default='deployment', help='Deployment directory')
    parser.add_argument('--mode', type=str, choices=['components', 'grade', 'tp53'], default='grade',
                        help='components or grade or tp53 output')

    args = parser.parse_args()

    X_raw = pd.read_csv(args.input, index_col=0).values
    pipe = CancerFeaturePipeline(deployment_dir=args.deployment_dir)

    if args.mode == 'components':
        components = pipe.predict_components(X_raw)
        comp_cols = [f'pred_component_{idx}' for idx in pipe.top_k_indices]
        out_df = pd.DataFrame(components, columns=comp_cols)
        out_df.to_csv(args.output, index=False)
        print(f"✓ Component predictions saved to {args.output}")

    elif args.mode == 'grade':
        results = pipe.predict_grade(X_raw)
        out_df = pd.DataFrame({
            'grade_prediction': results['grade_predictions'],
            'high_risk_probability': results['grade_probabilities'],
        })
        out_df.to_csv(args.output, index=False)
        print(f"✓ Grade predictions saved to {args.output}")

    else:
        results = pipe.predict_tp53(X_raw)
        out_df = pd.DataFrame({
            'tp53_prediction': results['tp53_predictions'],
            'tp53_probability': results['tp53_probabilities'],
        })
        out_df.to_csv(args.output, index=False)
        print(f"✓ TP53 predictions saved to {args.output}")
if __name__ == '__main__':

if __name__ == '__main__':
    main()
