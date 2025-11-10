"""
Visualize Results: 2D Heatmap and Analysis

This script creates visualizations for the comprehensive benchmark results:
- Heatmap of all embedding × explanation combinations
- Bar charts of top performers
- Comparison with exp-3 baseline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File paths
DETAILED_RESULTS = 'openrouter_results_all_explanations.csv'
EMBEDDING_SUMMARY = 'summary_by_embedding_model.csv'
LLM_SUMMARY = 'summary_by_explanation_llm.csv'
EXP3_RESULTS = '../exp-3/openrouter_results.csv'

# Output paths
HEATMAP_OUTPUT = 'heatmap_all_combinations.png'
TOP_EMBEDDINGS_OUTPUT = 'top_embedding_models.png'
TOP_LLMS_OUTPUT = 'top_explanation_llms.png'
COMPARISON_OUTPUT = 'comparison_with_exp3.png'


def load_data():
    """Load all required data files."""
    try:
        detailed = pd.read_csv(DETAILED_RESULTS)
        embedding_summary = pd.read_csv(EMBEDDING_SUMMARY)
        llm_summary = pd.read_csv(LLM_SUMMARY)
        print(f"✓ Loaded exp-5 results")

        exp3 = None
        if os.path.exists(EXP3_RESULTS):
            exp3 = pd.read_csv(EXP3_RESULTS)
            print(f"✓ Loaded exp-3 results for comparison")
        else:
            print(f"⚠ exp-3 results not found (skipping comparison)")

        return detailed, embedding_summary, llm_summary, exp3
    except FileNotFoundError as e:
        print(f"✗ Error: Required results file not found: {e}")
        print("Please run the benchmark first: python openrouter_benchmark_all_explanations.py")
        return None, None, None, None


def create_heatmap(detailed_df):
    """Create a 2D heatmap of all combinations."""
    print("\nGenerating heatmap...")

    # Pivot data to create matrix
    heatmap_data = detailed_df.pivot(
        index='Embedding Model',
        columns='Explanation Model',
        values='Accuracy (%)'
    )

    # Shorten model names for display
    heatmap_data.index = [name.split('/')[-1] for name in heatmap_data.index]
    heatmap_data.columns = [name.split('/')[-1][:20] for name in heatmap_data.columns]

    # Create figure
    plt.figure(figsize=(16, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=50,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Accuracy (%)'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title('Embedding Models × Explanation LLMs: Accuracy Heatmap', fontsize=16, pad=20)
    plt.xlabel('Explanation LLM Model', fontsize=12)
    plt.ylabel('Embedding Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(HEATMAP_OUTPUT, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved to: {HEATMAP_OUTPUT}")
    plt.close()


def create_top_embeddings_chart(embedding_summary_df):
    """Create bar chart of top embedding models."""
    print("\nGenerating top embeddings chart...")

    # Sort and get top models
    top_embeddings = embedding_summary_df.sort_values('Mean Accuracy', ascending=False)

    # Shorten names
    top_embeddings['Short Name'] = top_embeddings['Embedding Model'].apply(lambda x: x.split('/')[-1])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(
        top_embeddings['Short Name'],
        top_embeddings['Mean Accuracy'],
        color='steelblue',
        alpha=0.8
    )

    # Add error bars for std
    ax.errorbar(
        top_embeddings['Mean Accuracy'],
        range(len(top_embeddings)),
        xerr=top_embeddings['Std Accuracy'],
        fmt='none',
        color='black',
        alpha=0.5,
        capsize=5
    )

    # Add value labels
    for i, (mean, std) in enumerate(zip(top_embeddings['Mean Accuracy'], top_embeddings['Std Accuracy'])):
        ax.text(mean + 0.5, i, f'{mean:.1f}±{std:.1f}%', va='center', fontsize=9)

    ax.set_xlabel('Mean Accuracy (%) across all Explanation LLMs', fontsize=12)
    ax.set_ylabel('Embedding Model', fontsize=12)
    ax.set_title('Top Embedding Models (Mean ± Std)', fontsize=14, pad=15)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(TOP_EMBEDDINGS_OUTPUT, dpi=300, bbox_inches='tight')
    print(f"✓ Top embeddings chart saved to: {TOP_EMBEDDINGS_OUTPUT}")
    plt.close()


def create_top_llms_chart(llm_summary_df):
    """Create bar chart of top explanation LLMs."""
    print("\nGenerating top LLMs chart...")

    # Sort and get top models
    top_llms = llm_summary_df.sort_values('Mean Accuracy', ascending=False)

    # Shorten names
    top_llms['Short Name'] = top_llms['Explanation Model'].apply(lambda x: x.split('/')[-1][:25])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(
        top_llms['Short Name'],
        top_llms['Mean Accuracy'],
        color='coral',
        alpha=0.8
    )

    # Add error bars
    ax.errorbar(
        top_llms['Mean Accuracy'],
        range(len(top_llms)),
        xerr=top_llms['Std Accuracy'],
        fmt='none',
        color='black',
        alpha=0.5,
        capsize=5
    )

    # Add value labels
    for i, (mean, std) in enumerate(zip(top_llms['Mean Accuracy'], top_llms['Std Accuracy'])):
        ax.text(mean + 0.5, i, f'{mean:.1f}±{std:.1f}%', va='center', fontsize=9)

    ax.set_xlabel('Mean Accuracy (%) across all Embedding Models', fontsize=12)
    ax.set_ylabel('Explanation LLM Model', fontsize=12)
    ax.set_title('Top Explanation LLM Models (Mean ± Std)', fontsize=14, pad=15)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(TOP_LLMS_OUTPUT, dpi=300, bbox_inches='tight')
    print(f"✓ Top LLMs chart saved to: {TOP_LLMS_OUTPUT}")
    plt.close()


def create_comparison_chart(detailed_df, exp3_df):
    """Create comparison chart between exp-3 (raw poetry) and exp-5 (explanations)."""
    if exp3_df is None:
        print("\n⚠ Skipping comparison chart (exp-3 results not available)")
        return

    print("\nGenerating comparison chart...")

    # Calculate exp-5 average for each embedding model
    exp5_avg = detailed_df.groupby('Embedding Model')['Accuracy (%)'].mean().reset_index()
    exp5_avg.columns = ['Model Name', 'Exp-5 (Explanations)']

    # Merge with exp-3
    exp3_df_clean = exp3_df[['Model Name', 'Accuracy (%)']].copy()
    exp3_df_clean.columns = ['Model Name', 'Exp-3 (Raw Poetry)']

    comparison = pd.merge(exp5_avg, exp3_df_clean, on='Model Name', how='inner')

    if comparison.empty:
        print("⚠ No matching models between exp-3 and exp-5")
        return

    comparison['Delta'] = comparison['Exp-5 (Explanations)'] - comparison['Exp-3 (Raw Poetry)']
    comparison = comparison.sort_values('Delta', ascending=False)
    comparison['Short Name'] = comparison['Model Name'].apply(lambda x: x.split('/')[-1])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Side-by-side comparison
    x = np.arange(len(comparison))
    width = 0.35

    bars1 = ax1.bar(x - width/2, comparison['Exp-3 (Raw Poetry)'], width, label='Exp-3 (Raw Poetry)', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, comparison['Exp-5 (Explanations)'], width, label='Exp-5 (Explanations)', color='coral', alpha=0.8)

    ax1.set_xlabel('Embedding Model', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Raw Poetry vs Explanations: Accuracy Comparison', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison['Short Name'], rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)

    # Right plot: Delta (improvement)
    colors = ['green' if d > 0 else 'red' for d in comparison['Delta']]
    bars = ax2.barh(comparison['Short Name'], comparison['Delta'], color=colors, alpha=0.7)

    # Add value labels
    for i, delta in enumerate(comparison['Delta']):
        ax2.text(delta + 0.2 if delta > 0 else delta - 0.2, i, f'{delta:+.1f}%',
                va='center', ha='left' if delta > 0 else 'right', fontsize=9)

    ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_xlabel('Accuracy Delta (%)', fontsize=11)
    ax2.set_ylabel('Embedding Model', fontsize=11)
    ax2.set_title('Performance Change: Explanations vs Raw Poetry', fontsize=13)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(COMPARISON_OUTPUT, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison chart saved to: {COMPARISON_OUTPUT}")
    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    avg_exp3 = comparison['Exp-3 (Raw Poetry)'].mean()
    avg_exp5 = comparison['Exp-5 (Explanations)'].mean()
    avg_delta = comparison['Delta'].mean()

    improved = (comparison['Delta'] > 0).sum()
    degraded = (comparison['Delta'] < 0).sum()

    print(f"Average Accuracy (Exp-3 Raw Poetry):    {avg_exp3:.2f}%")
    print(f"Average Accuracy (Exp-5 Explanations):  {avg_exp5:.2f}%")
    print(f"Average Delta:                           {avg_delta:+.2f}%")
    print(f"\nModels improved with explanations:       {improved}/{len(comparison)}")
    print(f"Models degraded with explanations:       {degraded}/{len(comparison)}")

    if avg_delta > 0:
        print(f"\n✓ Explanations IMPROVE performance by {avg_delta:.2f}% on average")
    else:
        print(f"\n✗ Explanations DEGRADE performance by {abs(avg_delta):.2f}% on average")


def main():
    """Main visualization function."""
    print("="*60)
    print("Visualizing Benchmark Results")
    print("="*60)

    # Load data
    detailed, embedding_summary, llm_summary, exp3 = load_data()

    if detailed is None or embedding_summary is None or llm_summary is None:
        return

    # Create visualizations
    create_heatmap(detailed)
    create_top_embeddings_chart(embedding_summary)
    create_top_llms_chart(llm_summary)
    create_comparison_chart(detailed, exp3)

    print("\n" + "="*60)
    print("✓ All visualizations generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - {HEATMAP_OUTPUT}")
    print(f"  - {TOP_EMBEDDINGS_OUTPUT}")
    print(f"  - {TOP_LLMS_OUTPUT}")
    if exp3 is not None:
        print(f"  - {COMPARISON_OUTPUT}")


if __name__ == "__main__":
    main()
