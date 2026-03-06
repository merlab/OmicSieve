#!/usr/bin/env python
"""
Deployment inference script for cancer grade prediction.
Usage: python predict_grade.py --input <raw_expression.csv> --output <predictions.csv>
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
import phate
from sklearn.preprocessing import StandardScaler

class CancerGradePredictor:
    def __init__(self, deployment_dir='deployment'):
        self.deployment_dir = deployment_dir
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all saved models and configs."""
        self.scaler = joblib.load(os.path.join(self.deployment_dir, 'scaler.pkl'))
        self.phate_model = joblib.load(os.path.join(self.deployment_dir, 'supervised_phate.pkl'))
        self.feature_selector_mlp = torch.jit.load(os.path.join(self.deployment_dir, 'feature_selector_mlp.pt'))
        self.xgb_model = joblib.load(os.path.join(self.deployment_dir, 'xgboost_grade_predictor.pkl'))
        
        with open(os.path.join(self.deployment_dir, 'phate_supervision_config.json'), 'r') as f:
            self.phate_config = json.load(f)
        
        with open(os.path.join(self.deployment_dir, 'top_k_components.json'), 'r') as f:
            self.top_k_indices = json.load(f)
        
        with open(os.path.join(self.deployment_dir, 'component_gene_mapping.json'), 'r') as f:
            self.gene_mapping = json.load(f)
        
        print(f"✓ Loaded all artifacts from {self.deployment_dir}/")
    
    def predict_grade(self, X_raw, y_labels=None):
        """
        Predict cancer grades for new samples.
        
        Parameters:
        -----------
        X_raw : array-like of shape (n_samples, 19310)
            Raw gene expression matrix
        y_labels : array-like of shape (n_samples,), optional
            Binary labels (0=low risk, 1=high risk) for supervision enhancement.
            If None, assumes unlabeled (y=0 for all).
        
        Returns:
        --------
        predictions : dict
            'grade_predictions': array of 0 (low) or 1 (high)
            'grade_probabilities': array of probabilities for high-risk class
            'mlp_importance_scores': importance scores from MLP feature selector
            'top_components_used': indices of PHATE components used
            'interpretability': dict with gene contributions per sample
        """
        
        n_samples = X_raw.shape[0]
        
        # 1. Normalize with scaler
        X_normalized = self.scaler.transform(X_raw)
        
        # 2. Handle labels for supervision
        if y_labels is None:
            y_labels_onehot = np.zeros((n_samples, 2))
            y_labels_onehot[:, 0] = 1  # Default to label 0
        else:
            y_labels_onehot = np.zeros((n_samples, 2))
            y_labels_onehot[np.arange(n_samples), y_labels.astype(int)] = 1
        
        # 3. Concatenate labels for PHATE supervision
        supervision_weight = self.phate_config.get('supervision_weight', 5.0)
        X_with_labels = np.concatenate([X_normalized, y_labels_onehot * supervision_weight], axis=1)
        
        # 4. Transform with PHATE
        X_phate = self.phate_model.transform(X_with_labels)
        
        # 5. Select top-K components
        X_topk = X_phate[:, self.top_k_indices]
        
        # 6. Get importance scores from MLP
        X_topk_tensor = torch.FloatTensor(X_topk)
        with torch.no_grad():
            mlp_importance_scores = self.feature_selector_mlp(X_topk_tensor).numpy()
        
        # 7. Predict grades with XGBoost
        grade_predictions = self.xgb_model.predict(X_topk)
        grade_probabilities = self.xgb_model.predict_proba(X_topk)[:, 1]
        
        # 8. Build interpretability report
        interpretability = {}
        for i in range(min(3, n_samples)):  # Top 3 samples
            sample_genes = {}
            for comp_idx, top_k_idx in enumerate(self.top_k_indices[:5]):
                comp_key = f'component_{top_k_idx}'
                if comp_key in self.gene_mapping:
                    genes = self.gene_mapping[comp_key]['top_10_genes'][:3]
                    sample_genes[comp_key] = genes
            interpretability[f'sample_{i}'] = sample_genes
        
        return {
            'grade_predictions': grade_predictions,
            'grade_probabilities': grade_probabilities,
            'mlp_importance_scores': mlp_importance_scores,
            'top_components_used': self.top_k_indices,
            'interpretability_sample': interpretability,
            'n_samples': n_samples,
        }

def main():
    parser = argparse.ArgumentParser(description='Cancer Grade Prediction')
    parser.add_argument('--input', type=str, required=True, help='Input CSV with raw expression (rows=samples, cols=genes)')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV for predictions')
    parser.add_argument('--deployment-dir', type=str, default='deployment', help='Deployment directory')
    parser.add_argument('--labels', type=str, default=None, help='Optional labels CSV for supervision')
    
    args = parser.parse_args()
    
    # Load input
    X_raw = pd.read_csv(args.input, index_col=0).values
    y_labels = None
    if args.labels:
        y_labels = pd.read_csv(args.labels, index_col=0).values.flatten()
    
    predictor = CancerGradePredictor(deployment_dir=args.deployment_dir)
    
    results = predictor.predict_grade(X_raw, y_labels=y_labels)
    
    output_df = pd.DataFrame({
        'grade_prediction': results['grade_predictions'],
        'high_risk_probability': results['grade_probabilities'],
    })
    output_df.to_csv(args.output)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    main()
