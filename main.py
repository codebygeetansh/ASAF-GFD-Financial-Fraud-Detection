import os
import torch
import numpy as np
import pandas as pd
from utils.data_loader import load_elliptic_data
from utils.graph_filters import prepare_dual_path_features
from train import run_experiment

RESULTS_DIR = r"D:\BTech_Capstone_ASAF\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("=" * 65)
    print("🚀 STARTING RESEARCH PAPER EXPERIMENTS ON ELLIPTIC DATASET")
    print("=" * 65)

    # Step 1: Load Elliptic Dataset & Precomputed Dual-Path Features from D-Drive
    g, features, labels, train_mask, val_mask, test_mask = load_elliptic_data()
    Z1_bands, Z2_lowpass = prepare_dual_path_features(g, features, C=2, k=20)

    seeds = [42, 100, 2024, 777, 999]
    
    dpf_metrics = {'Rec@K': [], 'F1': [], 'AUC': [], 'AP': []}
    asaf_metrics = {'Rec@K': [], 'F1': [], 'AUC': [], 'AP': []}
    saved_artifacts = {}

    print("\n--- Running 5 Independent Trials (Base Paper DPF-GFD vs Proposed ASAF-GFD) ---")
    for idx, seed in enumerate(seeds, 1):
        print(f"\n[Run {idx}/5 | Seed = {seed}] Training models...")
        
        # 1. Base Paper Model (Static Concatenation)
        res_dpf, art_dpf = run_experiment(
            Z1_bands, Z2_lowpass, features, labels,
            train_mask, val_mask, test_mask,
            fusion_type='concat', seed=seed, epochs=100
        )
        for k in dpf_metrics:
            dpf_metrics[k].append(res_dpf[k])

        # 2. Tumhara Proposed Model (Adaptive Self-Attention Fusion)
        res_asaf, art_asaf = run_experiment(
            Z1_bands, Z2_lowpass, features, labels,
            train_mask, val_mask, test_mask,
            fusion_type='attention', seed=seed, epochs=100
        )
        for k in asaf_metrics:
            asaf_metrics[k].append(res_asaf[k])

        print(f"   DPF-GFD  (Static Concat) -> Rec@K: {res_dpf['Rec@K']:.4f} | F1: {res_dpf['F1']:.4f} | AUC: {res_dpf['AUC']:.4f} | AP: {res_dpf['AP']:.4f}")
        print(f"   ASAF-GFD (Ours - Attn)   -> Rec@K: {res_asaf['Rec@K']:.4f} | F1: {res_asaf['F1']:.4f} | AUC: {res_asaf['AUC']:.4f} | AP: {res_asaf['AP']:.4f}")

        if idx == 1:
            saved_artifacts['dpf'] = art_dpf
            saved_artifacts['asaf'] = art_asaf

    # Summary Table banana (Mean ± Std format - bilkul Base Paper Table 2 jaisa)
    print("\n" + "=" * 75)
    print("📊 FINAL EXPERIMENTAL RESULTS ON ELLIPTIC (MEAN ± STD OVER 5 RUNS)")
    print("=" * 75)
    print(f"{'Model':<28} | {'Rec@K':<15} | {'F1':<15} | {'AUC':<15} | {'AP':<15}")
    print("-" * 75)

    rows = []
    for name, m_dict in [("DPF-GFD (Static Concat)", dpf_metrics), ("ASAF-GFD (Proposed Attn)", asaf_metrics)]:
        rec_str = f"{np.mean(m_dict['Rec@K']):.4f}±{np.std(m_dict['Rec@K']):.4f}"
        f1_str  = f"{np.mean(m_dict['F1']):.4f}±{np.std(m_dict['F1']):.4f}"
        auc_str = f"{np.mean(m_dict['AUC']):.4f}±{np.std(m_dict['AUC']):.4f}"
        ap_str  = f"{np.mean(m_dict['AP']):.4f}±{np.std(m_dict['AP']):.4f}"
        print(f"{name:<28} | {rec_str:<15} | {f1_str:<15} | {auc_str:<15} | {ap_str:<15}")
        rows.append({'Model': name, 'Rec@K': rec_str, 'F1': f1_str, 'AUC': auc_str, 'AP': ap_str})

    print("=" * 75)

    # Fraud vs Normal nodes ke average Attention Weights print karo
    a1_test = saved_artifacts['asaf']['alpha_1_test']
    a2_test = saved_artifacts['asaf']['alpha_2_test']
    y_test = saved_artifacts['asaf']['y_test']

    print("\n🔍 ATTENTION WEIGHT ANALYSIS (TUMHARI NOVELTY KA PROOF):")
    print(f"   Fraud Nodes  -> Avg Structural Weight (alpha_1): {a1_test[y_test==1].mean():.4f} | Avg kNN Weight (alpha_2): {a2_test[y_test==1].mean():.4f}")
    print(f"   Normal Nodes -> Avg Structural Weight (alpha_1): {a1_test[y_test==0].mean():.4f} | Avg kNN Weight (alpha_2): {a2_test[y_test==0].mean():.4f}")

    # Save CSV and artifacts for Day 3 Plotting
    df_res = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "elliptic_comparison_results.csv")
    df_res.to_csv(csv_path, index=False)
    torch.save(saved_artifacts, os.path.join(RESULTS_DIR, "elliptic_plot_artifacts.pt"))
    print(f"\n✅ Results Table saved to: {csv_path}")
    print("✅ Plot Artifacts saved for Graphs & UMAP Visualization!")

if __name__ == "__main__":
    main()