"""
Compare Results: exp-3 (Raw Poetry) vs exp-5 (Explanations)

This script compares the performance of embedding models on raw Persian poetry
versus LLM-generated explanations of the poetry.
"""

import pandas as pd
import os
from pathlib import Path

# File paths
EXP3_RESULTS = '../exp-3/openrouter_results.csv'
EXP5_RESULTS = 'openrouter_results_explanations.csv'
COMPARISON_OUTPUT = 'comparison_raw_vs_explanations.csv'

def load_results(path: str, label: str) -> pd.DataFrame:
    """Load results CSV and add a source label."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)
    df['Source'] = label
    return df

def compare_results():
    """Generate comparison between exp-3 and exp-5 results."""

    print("="*80)
    print("Comparing Embedding Performance: Raw Poetry vs Explanations")
    print("="*80)

    # Load results from both experiments
    try:
        exp3_df = load_results(EXP3_RESULTS, 'Raw Poetry')
        print(f"\n✓ Loaded exp-3 results: {len(exp3_df)} models")
    except FileNotFoundError as e:
        print(f"\n✗ Error loading exp-3 results: {e}")
        print("  Please run exp-3 benchmark first!")
        return

    try:
        exp5_df = load_results(EXP5_RESULTS, 'Explanations')
        print(f"✓ Loaded exp-5 results: {len(exp5_df)} models")
    except FileNotFoundError as e:
        print(f"\n✗ Error loading exp-5 results: {e}")
        print("  Please run exp-5 benchmark first!")
        return

    # Merge results on model name
    merged = pd.merge(
        exp3_df[['Model Name', 'Accuracy (%)', 'Correct', 'Total']],
        exp5_df[['Model Name', 'Accuracy (%)', 'Correct', 'Total']],
        on='Model Name',
        suffixes=(' (Raw Poetry)', ' (Explanations)')
    )

    if merged.empty:
        print("\n✗ No common models found between exp-3 and exp-5!")
        return

    # Calculate delta
    merged['Delta (%)'] = merged['Accuracy (%) (Explanations)'] - merged['Accuracy (%) (Raw Poetry)']
    merged['Improvement'] = merged['Delta (%)'].apply(
        lambda x: '✓ Better' if x > 0 else ('✗ Worse' if x < 0 else '= Same')
    )

    # Sort by delta (highest improvement first)
    merged = merged.sort_values('Delta (%)', ascending=False)

    # Display comparison table
    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    print(merged.to_string(index=False))

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    avg_raw = merged['Accuracy (%) (Raw Poetry)'].mean()
    avg_exp = merged['Accuracy (%) (Explanations)'].mean()
    avg_delta = merged['Delta (%)'].mean()

    print(f"\nAverage Accuracy (Raw Poetry):    {avg_raw:.2f}%")
    print(f"Average Accuracy (Explanations):   {avg_exp:.2f}%")
    print(f"Average Delta:                     {avg_delta:+.2f}%")

    improved = (merged['Delta (%)'] > 0).sum()
    degraded = (merged['Delta (%)'] < 0).sum()
    unchanged = (merged['Delta (%)'] == 0).sum()

    print(f"\nModels improved with explanations:  {improved}/{len(merged)}")
    print(f"Models degraded with explanations:  {degraded}/{len(merged)}")
    print(f"Models unchanged:                   {unchanged}/{len(merged)}")

    # Best and worst performers
    best_idx = merged['Delta (%)'].idxmax()
    worst_idx = merged['Delta (%)'].idxmin()

    print("\n" + "="*80)
    print("NOTABLE RESULTS")
    print("="*80)

    print(f"\nMost Improved with Explanations:")
    print(f"  Model: {merged.loc[best_idx, 'Model Name']}")
    print(f"  Raw Poetry: {merged.loc[best_idx, 'Accuracy (%) (Raw Poetry)']:.2f}%")
    print(f"  Explanations: {merged.loc[best_idx, 'Accuracy (%) (Explanations)']:.2f}%")
    print(f"  Improvement: +{merged.loc[best_idx, 'Delta (%)']:.2f}%")

    print(f"\nMost Degraded with Explanations:")
    print(f"  Model: {merged.loc[worst_idx, 'Model Name']}")
    print(f"  Raw Poetry: {merged.loc[worst_idx, 'Accuracy (%) (Raw Poetry)']:.2f}%")
    print(f"  Explanations: {merged.loc[worst_idx, 'Accuracy (%) (Explanations)']:.2f}%")
    print(f"  Degradation: {merged.loc[worst_idx, 'Delta (%)']:.2f}%")

    # Best overall in each category
    best_raw_idx = merged['Accuracy (%) (Raw Poetry)'].idxmax()
    best_exp_idx = merged['Accuracy (%) (Explanations)'].idxmax()

    print(f"\nBest Model (Raw Poetry):")
    print(f"  Model: {merged.loc[best_raw_idx, 'Model Name']}")
    print(f"  Accuracy: {merged.loc[best_raw_idx, 'Accuracy (%) (Raw Poetry)']:.2f}%")

    print(f"\nBest Model (Explanations):")
    print(f"  Model: {merged.loc[best_exp_idx, 'Model Name']}")
    print(f"  Accuracy: {merged.loc[best_exp_idx, 'Accuracy (%) (Explanations)']:.2f}%")

    # Save comparison to CSV
    try:
        merged.to_csv(COMPARISON_OUTPUT, index=False)
        print("\n" + "="*80)
        print(f"✓ Comparison saved to: {COMPARISON_OUTPUT}")
        print("="*80)
    except Exception as e:
        print(f"\n✗ Error saving comparison: {e}")

    # Insights
    print("\n" + "="*80)
    print("INSIGHTS")
    print("="*80)

    if avg_delta > 0:
        print(f"\n✓ On average, explanations IMPROVE embedding performance by {avg_delta:.2f}%")
    elif avg_delta < 0:
        print(f"\n✗ On average, explanations DEGRADE embedding performance by {abs(avg_delta):.2f}%")
    else:
        print(f"\n= On average, explanations have NO EFFECT on embedding performance")

    if improved > degraded:
        print(f"✓ Explanations help more models ({improved}) than they hurt ({degraded})")
    elif degraded > improved:
        print(f"✗ Explanations hurt more models ({degraded}) than they help ({improved})")
    else:
        print(f"= Mixed results: equal number of improvements and degradations")

    print("\n" + "="*80)

if __name__ == "__main__":
    compare_results()
