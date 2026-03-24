#!/bin/bash
# Run multiple genetic algorithm simulations for one or more proteins,
# each with the same set of seeds, and generate fitness evolution plots.
# Also generates D2O% vs %D scatter plots for each generation.

set -e  # exit on error

# ---- Fixed parameters (can be changed here) ----
BATCH_SCRIPT="./parallel_process_pdb.sh"
PROCESSES=12
GENERATIONS=5
ELITISM=1
D2O_VAR=5
RATIO_THRESH=0.01

# Seeds to run (42 + 1..9)
SEEDS=(42 1 2 3 4 5 6 7 8 9)

# ------------------------------------------------

# Function to display usage
usage() {
    echo "Usage: $0 protein1 [protein2 ...]"
    echo "Example: $0 gfp mch rnase"
    echo "Each protein must have a corresponding PDB file: original/<protein>.pdb"
    exit 1
}

# Check if at least one protein is provided
if [ $# -eq 0 ]; then
    echo "ERROR: No protein names given."
    usage
fi

# Loop over all provided protein names
for PROTEIN in "$@"; do
    echo ""
    echo "========================================="
    echo "Processing protein: $PROTEIN"
    echo "========================================="

    INPUT_PDB="original/${PROTEIN}.pdb"
    if [ ! -f "$INPUT_PDB" ]; then
        echo "WARNING: PDB file $INPUT_PDB not found. Skipping $PROTEIN."
        continue
    fi

    BASE_DIR="test_result_${PROTEIN}/convergence_simulation"

    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- Running seed $SEED for $PROTEIN ---"

        # Define output directory for this seed
        OUT_DIR="${BASE_DIR}/seed_${SEED}/${PROTEIN}/"
        mkdir -p "$OUT_DIR"

        # Run the genetic algorithm
        python src/python_project/generate_deuterated_pdbs.py \
            "$INPUT_PDB" \
            --batch_script "$BATCH_SCRIPT" \
            -p "$PROCESSES" \
            -g "$GENERATIONS" \
            -e "$ELITISM" \
            --d2o-var "$D2O_VAR" \
            --seed "$SEED" \
            --ratio-threshold "$RATIO_THRESH" \
            --output_dir "$OUT_DIR"

        # Generate and save the fitness evolution plot
        CSV_FILE="${OUT_DIR}/best_fitness_summary.csv"
        PLOT_FILE="${OUT_DIR}/Fitness_evolution.png"

        if [ -f "$CSV_FILE" ]; then
            python src/python_project/plot_fitness_evolution.py \
                "$CSV_FILE" \
                --annotate \
                --min \
                -o "$PLOT_FILE"
            echo "Plot saved to $PLOT_FILE"
        else
            echo "Warning: $CSV_FILE not found – skipping plot."
        fi

        # Generate D2O% vs %D scatter plots for each generation
        python src/python_project/d2o_vs_d.py "$OUT_DIR"
        echo "D2O vs %D plots saved in ${OUT_DIR}/generation_plots_d2o_vs_d/"
    done
done

echo ""
echo "All simulations completed."
