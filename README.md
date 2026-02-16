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

#### Example:

- Command line:

```bash
python generate_deuterated_pdbs.py myprotein.pdb -p 30 -e 3 -g 5 --seed 42
```

- Using a config file:

```bash
python generate_deuterated_pdbs.py myprotein.pdb --batch_script ./process_pdb.sh >
```

### 2. Standalone PDB deuteration

Deuterate a single PDB file according to a specification:

```bash
python pdb_deuteration.py [config.ini] [options]
```

#### Examples:

- Command line:

```bash
python pdb_deuteration.py -i input.pdb -o output.pdb --d2o 50 --ALA --GLY
```
Flags for each amino acid (`--ALA`, `--GLY`, …) activate deuteration of that type.
Use `--all` to deuterate all amino acids.
`--no-ALA` etc. can be used to exclude an AA when another flag (e.g., `--all`) is active.

- Using a config file:

```bash
python pdb_deuteration.py pdb_config.ini
```
Command‑line arguments override values in the config file.

### 3. Standalone Fitness Evaluation

Evaluate fitness of existing `.dat` simulation files against references:

```bash
python fitness_evaluation.py <directory> [options]
```

`<directory>` must contain `.dat` files and a `ref/` subfolder with the two reference curves.
Outputs normalized fitness scores (one per line) and a summary.

####Options:

- `--q-max` : q truncation limit (default: 0.3 Å⁻¹)

- `--i0-threshold` : minimum I(0) ratio to pass filter (default: 0.2)

- `--deut-ref`, `--prot-ref` : specify custom reference filenames inside `ref/`.

## Configuration files

### `config.ini` : Genetic Algorithm and Fitness Parameters

Used by `generate_deuterated_pdbs.py` and `__init__.py` : 

- `[POPULATION]` : `population_size`, `elitism`, `d2o_variation_rate`
- `[GENETIC]` : `mutation_rate`, `crossover_rate`
- `[EXECUTION]` : `generations`, `seed`
- `[RESTRICTIONS]` : which amino acid types are modifiable by the GA (list of 20 booleans)
- `[FITNESS]` : `q_max`, `i0_threshold`, `deut_ref`, `prot_ref`

Command‑line arguments override values in the config file.

### `pdb_config.ini` : Standalone PDB deuteration

Used by `pdb_deuteration.py` :

- `[DEUTERATION]` : `input_pdb`, `output_pdb`, `d2o_percent`

- `[AMINO_ACIDS]` : which amino acid types are deuterated (list of 20 booleans)

