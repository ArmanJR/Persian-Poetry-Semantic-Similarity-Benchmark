"""
Visualization script for OpenRouter embedding benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Configuration
RESULTS_CSV_PATH = 'openrouter_results.csv'
OUTPUT_IMAGE_PATH = 'openrouter_results.png'


def plot_results(csv_path: str, output_path: str):
    """
    Generate a bar chart from benchmark results CSV.

    Args:
        csv_path: Path to the results CSV file
        output_path: Path to save the output chart
    """
    # Check if file exists
    if not Path(csv_path).exists():
        print(f"Error: Results file not found at '{csv_path}'")
        print("Please run openrouter_benchmark.py first to generate results.")
        sys.exit(1)

    # Load results
    try:
        results_df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"Loaded {len(results_df)} model results from {csv_path}")
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)

    # Validate required columns
    if 'Model Name' not in results_df.columns or 'Accuracy (%)' not in results_df.columns:
        print("Error: CSV file missing required columns 'Model Name' or 'Accuracy (%)'")
        sys.exit(1)

    # Sort by accuracy (descending)
    results_df = results_df.sort_values(by='Accuracy (%)', ascending=False).reset_index(drop=True)

    # Create figure
    plt.figure(figsize=(14, 8))

    # Create bar chart
    bars = plt.bar(
        range(len(results_df)),
        results_df['Accuracy (%)'],
        color='#4A90E2',
        alpha=0.8,
        edgecolor='black',
        linewidth=1.2
    )

    # Customize chart
    plt.xlabel("Embedding Model", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    plt.title(
        "OpenRouter Embedding Models - Persian Poetry Outlier Detection\nBenchmark Results",
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    plt.ylim(0, 105)

    # Set x-axis labels
    plt.xticks(
        range(len(results_df)),
        results_df['Model Name'],
        rotation=45,
        ha='right',
        fontsize=10
    )

    # Add accuracy values on top of bars
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1,
            f'{yval:.2f}%',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    # Add horizontal line at 25% for random baseline
    plt.axhline(
        y=25,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Random Selection Baseline (25%)',
        alpha=0.7
    )

    # Add legend
    plt.legend(loc='upper right', fontsize=10)

    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nChart successfully saved to: {output_path}")
    except Exception as e:
        print(f"Error saving chart: {e}")
        sys.exit(1)

    # Display chart
    print("Displaying chart...")
    plt.show()


def print_summary(csv_path: str):
    """Print a summary of the results."""
    results_df = pd.read_csv(csv_path, encoding='utf-8')
    results_df = results_df.sort_values(by='Accuracy (%)', ascending=False).reset_index(drop=True)

    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)

    if len(results_df) > 0:
        best_model = results_df.iloc[0]
        print(f"\nBest performing model: {best_model['Model Name']}")
        print(f"Accuracy: {best_model['Accuracy (%)']:.2f}%")

        if 'Correct' in results_df.columns and 'Total' in results_df.columns:
            print(f"Correct predictions: {best_model['Correct']}/{best_model['Total']}")

    print("\nBaseline (random selection): 25.00%")
    print()


if __name__ == "__main__":
    print("="*80)
    print("OpenRouter Embedding Results Visualization")
    print("="*80)

    # Print summary
    print_summary(RESULTS_CSV_PATH)

    # Generate chart
    print("\nGenerating visualization...")
    plot_results(RESULTS_CSV_PATH, OUTPUT_IMAGE_PATH)

    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)
