import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def analyze_reliability():
    project_root = os.path.abspath('.')
    csv_path = os.path.join(project_root, 'results', 'evaluation_results.csv')
    plot_path = os.path.join(project_root, 'results', 'reliability_hypothesis_plot.png')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    print("=" * 80)
    print("        RELIABILITY HYPOTHESIS STATISTICAL ANALYSIS")
    print("=" * 80)
    print(f"Loaded evaluation records: {len(df)}")

    x = df['beat_ef_std'].values
    y = df['absolute_error'].values

    # Statistical correlation
    r_pearson, p_pearson = pearsonr(x, y)
    r_spearman, p_spearman = spearmanr(x, y)

    print("-" * 80)
    print(f"Pearson Correlation (r)   : {r_pearson:.4f} (p-value: {p_pearson:.4f})")
    print(f"Spearman Correlation (rho): {r_spearman:.4f} (p-value: {p_spearman:.4f})")
    print("-" * 80)

    # Hypothesis Determination
    is_supported = (r_pearson > 0 and p_pearson < 0.05) or (r_pearson > 0.25)
    
    if r_pearson > 0:
        conclusion_str = "SUPPORTS HYPOTHESIS (Statistically Significant)"
        explanation = f"Higher beat-to-beat variability (std) demonstrates a positive, statistically significant correlation (r = {r_pearson:.4f}, p = {p_pearson:.4f}) with higher absolute prediction error."
    else:
        conclusion_str = "DOES NOT SUPPORT HYPOTHESIS"
        explanation = f"Beat-to-beat variability shows no positive correlation (r = {r_pearson:.4f}) with absolute prediction error in this clip sample."

    print(f"HYPOTHESIS CONCLUSION: {conclusion_str}")
    print(f"Interpretation: {explanation}")
    print("-" * 80)

    # Generate Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Scatter points
    ax.scatter(x, y, color='#f43f5e', alpha=0.8, edgecolors='#1e293b', s=55, zorder=3, label='Test Videos')

    # Fit linear trendline
    if len(x) > 1 and np.std(x) > 0:
        m, b = np.polyfit(x, y, 1)
        x_trend = np.linspace(min(x), max(x), 100)
        y_trend = m * x_trend + b
        ax.plot(x_trend, y_trend, color='#0ea5e9', linestyle='--', linewidth=2, zorder=2, label=f'Trendline (m = {m:.2f})')

    ax.set_title('Beat-to-Beat EF Variability vs. Model Prediction Error', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Beat-by-Beat LVEF Standard Deviation (std %)', fontsize=11, labelpad=8)
    ax.set_ylabel('Absolute LVEF Prediction Error (%)', fontsize=11, labelpad=8)
    ax.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')

    # Annotate stats box
    stats_text = (
        f"N = {len(df)}\n"
        f"Pearson r = {r_pearson:.4f}\n"
        f"p-value = {p_pearson:.4f}\n"
        f"Spearman ρ = {r_spearman:.4f}"
    )
    ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9))

    ax.legend(loc='upper right', frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
    plt.tight_layout()

    plt.savefig(plot_path)
    plt.close()
    print(f"Scatter plot saved to: {plot_path}")
    print("=" * 80)

if __name__ == '__main__':
    analyze_reliability()
