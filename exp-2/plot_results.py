import pandas as pd # For saving to CSV
import matplotlib.pyplot as plt # For plotting the chart
import matplotlib as mpl

# Custom dark theme settings
custom_dark_theme = {
    "figure.facecolor": "#111111",
    "axes.facecolor": "#111111",
    "axes.edgecolor": "#dddddd",
    "axes.labelcolor": "#dddddd",
    "xtick.color": "#dddddd",
    "ytick.color": "#dddddd",
    "text.color": "#ffffff",
    "axes.titleweight": "bold",
    "axes.titlepad": 15.0,
    "axes.grid": True,
    "grid.color": "#444444",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "legend.facecolor": "#222222",
    "legend.edgecolor": "#dddddd",
    "savefig.facecolor": "#111111",
    "savefig.edgecolor": "#111111",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

# Apply custom theme
mpl.rcParams.update(custom_dark_theme)

results_df = pd.read_csv("benchmark_results.csv")

plt.figure(figsize=(16, 6))
bars = plt.bar(results_df['Model Name'], results_df['Accuracy (%)'], color='skyblue')

plt.xlabel("Model Name")
plt.ylabel("Accuracy (%)")
plt.title("Persian Poetry Outlier Detection Benchmark Results")
plt.ylim(0, 105) # Set Y-axis limit slightly above 100%
plt.xticks(rotation=30, ha='right') # Rotate labels to prevent overlap

# Add accuracy values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', va='bottom', ha='center') # Adjust position slightly

# Add horizontal line at 25% to indicate random selection
plt.axhline(y=25, color='red', linestyle='--', linewidth=1, label='Random Selection (25%)')
plt.legend()

plt.tight_layout()

# Display the chart
plt.show()