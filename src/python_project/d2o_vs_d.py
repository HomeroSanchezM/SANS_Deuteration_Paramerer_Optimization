#!/usr/bin/env python3
"""
Generate scatter plots of D₂O% vs %D from generation summary files.
Points are coloured by fitness (0 → max fitness of the generation).
The best individual is outlined in red.

Usage: python plot_generations_d2o_vs_d.py <input_directory>
"""

import os
import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def parse_generation_file(filepath):
    """
    Parse a generation summary file and return generation number + data rows.
    Each row contains (d2o_percent, d_percent, fitness).
    """
    generation_num = None
    rows = []

    with open(filepath, "r") as f:
        content = f.read()

    # Extract generation number from header
    match = re.search(r"POPULATION SUMMARY\s*[-]+\s*GENERATION\s+(\d+)", content)
    if match:
        generation_num = int(match.group(1))
    else:
        # Fallback: extract from filename (e.g. generation_01_summary.txt)
        fname = os.path.basename(filepath)
        match = re.search(r"(\d+)", fname)
        if match:
            generation_num = int(match.group(1))

    # Detect whether the file contains a "ratio" column
    has_ratio = bool(re.search(r"\bratio\b", content, re.IGNORECASE))

    # Regular expressions to capture D2O%, AA_deut, Fitness, and the following %D column.
    # After %D we ignore any remaining columns (Non_labile_D%, Created, etc.)
    if has_ratio:
        # Format: Rank PDB D2O AA ratio Fitness D% ...
        row_pattern = re.compile(
            r"^\s*\d+\s+\S+\s+(\d+)\s+(\d+)\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+\S.*$"
        )
    else:
        # Format: Rank PDB D2O AA Fitness D% ...
        row_pattern = re.compile(
            r"^\s*\d+\s+\S+\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+\S.*$"
        )

    for line in content.splitlines():
        m = row_pattern.match(line)
        if m:
            d2o = int(m.group(1))          # D₂O% (integer)
            # aa_deut = int(m.group(2))     # not used here
            fitness = float(m.group(3))
            d_percent = float(m.group(4))   # %D (float)
            rows.append((d2o, d_percent, fitness))

    return generation_num, rows


def plot_generation(generation_num, rows, output_dir):
    """Create and save a scatter plot for one generation."""
    if not rows:
        print(f"  No data found for Generation {generation_num}, skipping.")
        return

    d2o_vals = np.array([r[0] for r in rows])
    d_vals   = np.array([r[1] for r in rows])
    fitness_vals = np.array([r[2] for r in rows])

    # Identify best fitness point(s)
    max_fitness = np.max(fitness_vals)
    best_indices = np.where(fitness_vals == max_fitness)[0]

    # Transparency: fitness 0 → nearly transparent
    alphas = np.where(fitness_vals == 0, 0.05, 0.9)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Colour map: plasma, scaled from 0 to max fitness of this generation
    cmap = cm.get_cmap("plasma")
    scatter = ax.scatter(
        d2o_vals,
        d_vals,
        c=fitness_vals,
        cmap=cmap,
        s=80,
        edgecolors="black",
        linewidths=0.5,
        alpha=alphas,
        vmin=0,
        vmax=max_fitness,
    )

    # Highlight best individual(s) with a red circle
    for idx in best_indices:
        ax.scatter(
            d2o_vals[idx],
            d_vals[idx],
            s=140,
            facecolors="none",
            edgecolors="red",
            linewidths=2.5,
            zorder=5
        )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Fitness", fontsize=11)

    ax.set_xlabel("D₂O (%)", fontsize=12)
    ax.set_ylabel("Total deuteration %D", fontsize=12)
    ax.set_title(f"Generation {generation_num:02d} — D₂O% vs %D", fontsize=14, fontweight="bold")

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-1, 101)
    ax.set_ylim(0, 100)          # %D typically between 0 and 100

    plt.tight_layout()

    out_path = os.path.join(output_dir, f"generation_{generation_num:02d}_d2o_vs_d.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_generations_d2o_vs_d.py <input_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a valid directory.")
        sys.exit(1)

    output_dir = os.path.join(input_dir, "generation_plots_d2o_vs_d")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Find all summary files (flexible naming)
    patterns = ["*generation*summary*.txt", "*gen*summary*.txt", "*summary*.txt"]
    files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(input_dir, pattern))
        if files:
            break

    if not files:
        print("No summary files found. Looking for any .txt files...")
        files = glob.glob(os.path.join(input_dir, "*.txt"))

    if not files:
        print("No .txt files found in the directory.")
        sys.exit(1)

    files.sort()
    print(f"Found {len(files)} file(s) to process.\n")

    for filepath in files:
        print(f"Processing: {os.path.basename(filepath)}")
        generation_num, rows = parse_generation_file(filepath)
        if generation_num is None:
            print("  Could not determine generation number, skipping.")
            continue
        print(f"  Generation {generation_num}: {len(rows)} individuals parsed.")
        plot_generation(generation_num, rows, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()