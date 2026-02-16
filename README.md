# Deuteration Optimization using Genetic Algorithm for SANS

## Overview

This project implements a genetic algorithm (GA) to find the optimal deuteration pattern of a protein for Small-Angle Neutron Scattering (SANS) contrast variation experiments.

The algorithm evolves a population of chromosomes, each representing:

- Which amino acid types are fully deuterated (non‑labile H → D)

- The D₂O percentage of the solvent (labile H → D exchange)

Each chromosome is used to generate a deuterated PDB file, which is then passed to a SANS simulation program (e.g., Pepsi‑SANS). The fitness of a solution is defined as the sum of the areas between the scaled simulated curve and two experimental references (fully deuterated and fully protonated). A larger area (after normalization) corresponds to a better fit, i.e., a higher fitness score (closer to 1).

The main script `generate_deuterated_pdbs.py` orchestrates the entire workflow, including calling the external simulation script `process_pdb.sh` which runs Pepsi‑SANS on all generated PDB files.

All components can also be used independently as command‑line tools.

## Requirements

- `Pepsi-SANS` =3.0 ( Linux executable in `./Pepsi-SANS-Linux/Pepsi-SANS` )
- `gromacs` >=2026.0<2027
- `Python` >= 3.11
- Python packages : 
	- `numpy` >=2.4.1<3
	- `dataclasses` >=0.8<0.9
	- `biopython` >=1.86<2
	- `scipy` >=1.17.0<2
	- `gemmi` >=0.7.4<0.8

## Usage 

### 1. Main script : Full genetic algorithm workflow 

```bash
python generate_deuterated_pdbs.py <input.pdb> [options]

```
#### Required positional argument:

`input.pdb`: the non-deuterated protein strcuture (all H have to be explicit and protonated).

#### Common options:

| Option | Description |
|:-----------|:-----------|
| `-p, --population_size` | Population size (must be multiple of 3) |
| `-e, --elitism` | Number of elite individuals preserved. `≤population_size/3` |
| `-v, --d2o_variation_rate` | Max D₂O change in mutation (±). |
| `-m, --mutation_rate` | Mutation probability (0–1). |
| `-c, --crossover_rate` | Crossover probability (0–1). |
| `-g, --generations` | Number of generations. |
| `--seed` | Random seed for reproducibility. |
| `--output_dir` | Output folder. Default: `<pdb_basename>_deuterated_pdbs` |
| `--batch_script` | Path to the batch processing script. Default: `./process_pdb.sh` |

