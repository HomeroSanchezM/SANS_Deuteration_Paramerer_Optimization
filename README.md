# Deuteration Optimization using Genetic Algorithm for SANS

## Overview

This project implements a genetic algorithm (GA) to find the optimal deuteration pattern of a protein for Small-Angle Neutron Scattering (SANS) contrast variation experiments.

The algorithm evolves a population of chromosomes, each representing:

- Which amino acid types are fully deuterated (non‑labile H → D)

- The D₂O percentage of the solvent (labile H → D exchange)

Each chromosome is used to generate a deuterated PDB file, which is then passed to a SANS simulation program (e.g., Pepsi‑SANS). The fitness of a solution is defined as the sum of the areas between the scaled simulated curve and two experimental references (fully deuterated and fully protonated). A larger area (after normalization) corresponds to a better fit, i.e., a higher fitness score (closer to 1).

The main script `generate_deuterated_pdbs.py` orchestrates the entire workflow, including calling the external simulation script `process_pdb.sh` which runs Pepsi‑SANS on all generated PDB files.

All components can also be used independently as command‑line tools.
