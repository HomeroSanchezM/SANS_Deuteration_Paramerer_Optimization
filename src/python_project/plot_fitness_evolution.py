#!/usr/bin/env python3
"""
Plot Fitness Evolution across Generations
==========================================

Reads the best_fitness_summary.csv produced by generate_deuterated_pdbs.py
and creates a figure showing:
  - Best fitness (area difference) per generation  [left y-axis, blue]
  - D2O percentage of the best chromosome          [right y-axis, orange dashed]

The number of deuterated AAs is annotated above each fitness point.
With --annotate, a colour grid is drawn below the plot: each column is a
generation, each row is one of the 20 standard AAs (fixed order). Cells are
coloured green (deuterated) or red (not deuterated). Exact fitness, %D and
%Non-labile-D values are shown below the grid.

Usage:
  python plot_fitness_evolution.py best_fitness_summary.csv
  python plot_fitness_evolution.py best_fitness_summary.csv -o fitness_plot.png
  python plot_fitness_evolution.py best_fitness_summary.csv --annotate --min
  python plot_fitness_evolution.py best_fitness_summary.csv --interactive   # show plot
"""

import sys
import argparse
import csv
from pathlib import Path

# Canonical AA order (matches image legend)
AMINO_ACIDS = [
    ("Alanine",       "ALA", "A"),
    ("Arginine",      "ARG", "R"),
    ("Asparagine",    "ASN", "N"),
    ("Aspartic acid", "ASP", "D"),
    ("Cysteine",      "CYS", "C"),
    ("Glutamic acid", "GLU", "E"),
    ("Glutamine",     "GLN", "Q"),
    ("Glycine",       "GLY", "G"),
    ("Histidine",     "HIS", "H"),
    ("Isoleucine",    "ILE", "I"),
    ("Leucine",       "LEU", "L"),
    ("Lysine",        "LYS", "K"),
    ("Methionine",    "MET", "M"),
    ("Phenylalanine", "PHE", "F"),
    ("Proline",       "PRO", "P"),
    ("Serine",        "SER", "S"),
    ("Threonine",     "THR", "T"),
    ("Tryptophan",    "TRP", "W"),
    ("Tyrosine",      "TYR", "Y"),
    ("Valine",        "VAL", "V"),
]
AA_CODES = [aa[1] for aa in AMINO_ACIDS]        # 3-letter codes in order
AA_SHORT = [aa[0][:3].capitalize()              # Short display names (Ala, Arg …)
            if aa[0][:3].lower() != 'asp' else 'Asp'
            for aa in AMINO_ACIDS]
# Build nicer short labels matching the image exactly
AA_LABELS = [
    "Ala","Arg","Asn","Asp","Cys",
    "Glu","Gln","Gly","His","Ile",
    "Leu","Lys","Met","Phe","Pro",
    "Ser","Thr","Trp","Tyr","Val",
]

COLOR_GREEN = "#4CAF50"
COLOR_RED   = "#E53935"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot best fitness evolution across genetic algorithm generations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('summary_csv', type=str,
                        help='Path to best_fitness_summary.csv')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output image path (e.g. plot.png). If omitted and not interactive, saves to Fitness_evolution.png beside the CSV.')
    parser.add_argument('--title', type=str, default='Best Fitness per Generation',
                        help='Plot title')
    parser.add_argument('--annotate', action='store_true',
                        help='Add colour grid of deuterated AAs below the plot')
    parser.add_argument('--min', action='store_true',
                        help='With --annotate: show stat values only on change and at last column')
    parser.add_argument('--interactive', action='store_true',
                        help='Show the plot interactively (instead of saving to file)')
    return parser.parse_args()


def load_summary(csv_path):
    generations, fitness_values, d2o_values = [], [], []
    n_deut_values, aa_lists = [], []
    pct_d_values, pct_nonlabile_values, ratio_values = [], [], []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append(int(row['generation']))
            fitness_values.append(float(row['fitness']))
            d2o_values.append(int(row['d2o_percent']))
            n_deut_values.append(int(row['n_deuterated_aa']))
            aa_lists.append(set(row['deuterated_aa_list'].split(';')) if row['deuterated_aa_list'] else set())
            pct_d_values.append(float(row['%D']))
            pct_nonlabile_values.append(float(row['%Non_labile_D%']))
            ratio_values.append(float(row['ratio']))

    if not generations:
        raise ValueError("No data rows found in summary CSV.")

    return generations, fitness_values, d2o_values, n_deut_values, aa_lists, pct_d_values, pct_nonlabile_values, ratio_values


def build_aa_matrix(aa_lists):
    """Return a (20 x n_gen) integer matrix: 1=deuterated, 0=not."""
    n_gen = len(aa_lists)
    matrix = []
    for code in AA_CODES:
        row = [1 if code in aa_set else 0 for aa_set in aa_lists]
        matrix.append(row)
    return matrix   # shape: [20][n_gen]


def main():
    args = parse_arguments()

    try:
        import matplotlib
        # Decide backend: if not interactive and we have an output, use Agg (no display)
        if not args.interactive:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError as e:
        print(f"ERROR: {e}. Install with: pip install matplotlib numpy")
        sys.exit(1)

    csv_path = Path(args.summary_csv)
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    try:
        (generations, fitness_values, d2o_values, n_deut_values,
         aa_lists, pct_d_values, pct_nonlabile_values, ratio_values) = load_summary(str(csv_path))
    except (ValueError, KeyError) as e:
        print(f"ERROR reading summary CSV: {e}")
        sys.exit(1)

    n_gen = len(generations)
    color_fit = '#1565C0'
    color_d2o = '#E65100'

    # Determine output filename if not given
    if args.output is None and not args.interactive:
        # Default: place Fitness_evolution.png next to the input CSV
        args.output = str(csv_path.parent / "Fitness_evolution.png")
        print(f"No output specified, saving to: {args.output}")

    # ------------------------------------------------------------------ #
    #  Layout: one panel (curve only) or two panels (curve + AA grid)     #
    # ------------------------------------------------------------------ #
    fig_width = max(10, n_gen * 0.9 + 3)

    if not args.annotate:
        fig, ax1 = plt.subplots(figsize=(fig_width, 5))
    else:
        N_AA   = len(AA_CODES)   # 20
        N_STAT = 3               # fitness, %D, %Non_labile_D
        # heights: curve panel is taller; grid panel height proportional to rows
        grid_height_ratio = (N_AA + N_STAT + 1) * 0.22   # in inches per row
        fig = plt.figure(figsize=(fig_width, 5 + grid_height_ratio))
        gs = gridspec.GridSpec(
            2, 1,
            height_ratios=[5, grid_height_ratio],
            hspace=0.08,
        )
        ax1 = fig.add_subplot(gs[0])

    # ------------------------------------------------------------------ #
    #  Top panel: fitness curve                                           #
    # ------------------------------------------------------------------ #
    line_fit, = ax1.plot(
        generations, fitness_values,
        marker='o', color=color_fit, linewidth=2, markersize=7,
        label='Best fitness (area diff.)'
    )
    ax1.set_ylabel('Fitness (area difference)', color=color_fit, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_fit)
    ax1.set_title(args.title, fontsize=14, fontweight='bold')
    ax1.set_xticks(generations)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    if not args.annotate:
        ax1.set_xlabel('Generation', fontsize=12)

    ax2 = ax1.twinx()
    line_d2o, = ax2.plot(
        generations, d2o_values,
        marker='s', color=color_d2o, linewidth=2, linestyle='--', markersize=6,
        label='D\u2082O %'
    )
    ax2.set_ylabel('D\u2082O (%)', color=color_d2o, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_d2o)
    ax2.set_ylim(-5, 110)

    # Annotate n_deuterated_aa above each fitness point
    for g, f, n in zip(generations, fitness_values, n_deut_values):
        ax1.annotate(
            f'{n}',
            xy=(g, f), xytext=(0, 8), textcoords='offset points',
            ha='center', va='bottom', fontsize=8, color=color_fit
        )

    lines  = [line_fit, line_d2o]
    labels = [l.get_label() for l in lines]

    # Legend combining both axes — only shown when not annotating (annotate mode rebuilds it)
    if not args.annotate:
        ax1.legend(lines, labels, loc='upper left', fontsize=10)

    # ------------------------------------------------------------------ #
    #  Bottom panel: AA colour grid                                       #
    # ------------------------------------------------------------------ #
    if args.annotate:
        ax_grid = fig.add_subplot(gs[1])
        ax_grid.set_xlim(-1, n_gen)
        ax_grid.set_ylim(-N_STAT - 0.5, N_AA + 0.5)
        ax_grid.axis('off')

        matrix = build_aa_matrix(aa_lists)

        cell_w = 1.0   # each cell is 1 unit wide in data coords
        cell_h = 1.0

        # Draw AA label column + coloured cells
        for row_i, (label, code) in enumerate(zip(AA_LABELS, AA_CODES)):
            y = N_AA - 1 - row_i   # top-down

            # AA name label (left side, at x = -0.55)
            ax_grid.text(
                -0.55, y + 0.5, label,
                ha='right', va='center', fontsize=7.5, fontweight='bold'
            )

            # Coloured rectangles for each generation
            for col_i in range(n_gen):
                is_deut = matrix[row_i][col_i]
                color   = COLOR_GREEN if is_deut else COLOR_RED
                rect = mpatches.FancyBboxPatch(
                    (col_i + 0.05, y + 0.05),
                    cell_w - 0.10, cell_h - 0.10,
                    boxstyle="round,pad=0.02",
                    linewidth=0,
                    facecolor=color,
                )
                ax_grid.add_patch(rect)

        # ---- Stats rows below the AA grid ----
        STAT_LABELS = ['Fitness', '%D', '%Non-lab.D', 'Ratio']
        STAT_RAW    = [fitness_values, pct_d_values, pct_nonlabile_values, ratio_values]
        STAT_FMT    = ['{:.5f}', '{:.1f}', '{:.1f}', '{:.3f}']
        STAT_COLORS = ['#1565C0', '#2E7D32', '#6A1B9A', '#B71C1C']

        # Separator line
        ax_grid.axhline(y=-0.5, xmin=0, xmax=1, color='#555', linewidth=0.8,
                        linestyle='--', alpha=0.7)

        last_col = n_gen - 1

        for s_i, (slabel, sraw, sfmt, scol) in enumerate(zip(STAT_LABELS, STAT_RAW, STAT_FMT, STAT_COLORS)):
            y_s = -1 - s_i   # place below the AA grid

            # Row label
            ax_grid.text(
                -0.55, y_s + 0.5, slabel,
                ha='right', va='center', fontsize=7, fontweight='bold',
                color=scol
            )

            # Values per generation — respect --min flag
            for col_i, val in enumerate(sraw):
                is_last   = (col_i == last_col)
                is_change = (col_i == 0) or (val != sraw[col_i - 1])
                show = (not args.min) or is_change or is_last
                if show:
                    ax_grid.text(
                        col_i + 0.5, y_s + 0.5, sfmt.format(val),
                        ha='center', va='center', fontsize=6,
                        color=scol, rotation=0
                    )

        # Generation tick labels under the grid (align with curve x-axis)
        for col_i, g in enumerate(generations):
            ax_grid.text(
                col_i + 0.5, N_AA + 0.05, str(g),
                ha='center', va='bottom', fontsize=6.5, color='#333'
            )
        ax_grid.text(
            -0.55, N_AA + 0.05, 'Gen',
            ha='right', va='bottom', fontsize=6.5, color='#555', style='italic'
        )

        # Sync x-range of top axis with grid column positions
        # Map generation index -> data coord for the curve
        # The grid cells span [0, n_gen] in data coords; align top ax accordingly
        ax1.set_xlim(generations[0] - 0.5, generations[-1] + 0.5)
        ax2.set_xlim(generations[0] - 0.5, generations[-1] + 0.5)

        legend_patches = [
            mpatches.Patch(color=COLOR_GREEN, label='Deuterated'),
            mpatches.Patch(color=COLOR_RED,   label='Not deuterated'),
        ]
        ax1.legend(
            handles=lines + legend_patches,
            labels=labels + ['Deuterated', 'Not deuterated'],
            loc='lower right',
            fontsize=9,
            framealpha=0.9,
        )

    plt.tight_layout()

    if args.interactive:
        plt.show()
    else:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()