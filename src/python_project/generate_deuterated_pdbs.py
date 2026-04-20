#!/usr/bin/env python3
"""
Deuterated PDB Generator with Genetic Algorithm
================================================

Main script to generate a population of deuteration patterns,
create corresponding deuterated PDBs, run SANS simulation,
and evaluate fitness automatically.

By default, two reference PDBs are created in ref/:
  - <protein>_total_deuteration.pdb  (protonated in D2O)
  - <protein>_total_protonation.pdb  (protonated in H2O)

Usage examples:
  python generate_deuterated_pdbs.py input.pdb
  python generate_deuterated_pdbs.py input.pdb --no_default_ref --ref ref1.pdb ref2.pdb
  python generate_deuterated_pdbs.py input.pdb --ref extra_ref.pdb
  python generate_deuterated_pdbs.py input.pdb -p 60 -e 5 -g 20 --seed 123
"""

import os
import sys
import argparse
import random
import logging
import subprocess
import configparser
import tempfile
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import numpy as np

from __init__ import (
    AMINO_ACIDS,
    EFFECTIVE_AMINO_ACIDS,       # 18 effective genes
    N_EFFECTIVE_AA,
    AA_GROUP_INDEX,          
    expand_deuteration_vector,
    merge_restrictions_to_18,
    Chromosome,
    PopulationGenerator,
    restrictions as default_restrictions,
)
from pdb_deuteration import PdbDeuteration
from fitness_evaluation import evaluate_population_fitness

# ============================================================================
#                           LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# AA display labels matching canonical AMINO_ACIDS order (used for plots)
_AA_PLOT_LABELS = [
    "Ala", "Arg", "Asn", "Asp", "Cys",
    "Glu", "Gln", "Gly", "His", "Ile",
    "Leu", "Lys", "Met", "Phe", "Pro",
    "Ser", "Thr", "Trp", "Tyr", "Val",
]


# ============================================================================
#                           FILENAME HELPERS
# ============================================================================

def get_pdb_filename(chrom: Chromosome) -> str:
    """
    Derive the PDB filename from the chromosome's intrinsic attributes.

    Format: gen{generation:02d}_Chr{index:03d}_d2o{d2o:03d}_deut{n_deut:02d}.pdb

    ``n_deut`` is the number of deuterated effective genes (0-18).  Because
    the linked-pair scheme uses 18 genes, this value can now be at most 18
    instead of 20 in the old per-AA scheme — filenames remain uniquely
    descriptive.
    """
    num_deut = sum(chrom.deuteration)   # count of True genes in 18-element vector
    return (f"gen{chrom.generation:02d}_Chr{chrom.index:03d}"
            f"_d2o{chrom.d2o:03d}_deut{num_deut:02d}.pdb")


def get_sorted_indices(population: List[Chromosome]) -> List[int]:
    """
    Return the indices into *population* sorted by descending fitness.

    Usage:
        sorted_indices = get_sorted_indices(population)
        best_chrom = population[sorted_indices[0]]
    """
    return sorted(
        range(len(population)),
        key=lambda i: population[i].fitness if population[i].fitness is not None else 0.0,
        reverse=True
    )


# ============================================================================
#                           ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command-line arguments for the genetic algorithm workflow."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate deuterated PDBs using a genetic algorithm "
            "and evaluate fitness via Pepsi-SANS"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # ---- Positional ----
    parser.add_argument('pdb_file', type=str, help='Source PDB file to deuterate')

    # ---- Config file ----
    parser.add_argument('--config', type=str, metavar='FILE',
                        help='Configuration INI file (CLI arguments override INI)')

    # ---- I/O ----
    io = parser.add_argument_group('Input/Output')
    io.add_argument('--output_dir', type=str,
                    help='Output directory (default: <pdb_basename>_deuterated_pdbs)')
    io.add_argument('--batch_script', type=str, default='./process_pdb.sh',
                    help='Path to the batch processing script')

    ref = parser.add_argument_group('Reference PDBs')
    ref.add_argument('--no_default_ref', action='store_true',
                     help='Do not create the default protonated-in-D2O / H2O reference PDBs')
    ref.add_argument('--ref', type=str, nargs='+', default=None, metavar='PDB',
                     help='Additional reference PDB file(s) to copy into ref/ and use for fitness. '
                          'Can be used with or without --no_default_ref.')

    # ---- Population ----
    pop = parser.add_argument_group('Population parameters')
    pop.add_argument('-p', '--population_size', type=int)
    pop.add_argument('-e', '--elitism', type=int)
    pop.add_argument('--d2o-var', '--d2o_variation_rate', dest='d2o_variation_rate', type=int)
    pop.add_argument('--d2o', type=int, nargs='+', default=None, metavar='VALUE')

    # ---- Execution ----
    exc = parser.add_argument_group('Execution parameters')
    exc.add_argument('-g', '--generations', type=int)
    exc.add_argument('--seed', type=int)

    # ---- Fitness ----
    fit = parser.add_argument_group('Fitness evaluation parameters')
    fit.add_argument('--q-max', type=float)
    fit.add_argument('--ratio-threshold', type=float)
    fit.add_argument('--i0-threshold', type=float, help=argparse.SUPPRESS)
    fit.add_argument('--deut-ref', type=str, help=argparse.SUPPRESS)  # deprecated
    fit.add_argument('--prot-ref', type=str, help=argparse.SUPPRESS)  # deprecated

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    return args


# ============================================================================
#                     CONFIGURATION LOADING & MERGING
# ============================================================================

def load_config_ini(path: str) -> Dict[str, Any]:
    """
    Load genetic algorithm configuration from an INI file.

    The [RESTRICTIONS] section is expected to contain 20 boolean entries
    (one per canonical AA).  This function automatically converts them to the
    18-element effective restriction vector used by the chromosome, applying
    OR logic for linked pairs (ASN+ASP, GLU+GLN).
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = configparser.ConfigParser()
    config.read(path)
    cfg: Dict[str, Any] = {}

    if config.has_section("POPULATION"):
        cfg["population_size"]   = config.getint("POPULATION", "population_size", fallback=None)
        cfg["elitism"]           = config.getint("POPULATION", "elitism", fallback=None)
        cfg["d2o_variation_rate"]= config.getint("POPULATION", "d2o_variation_rate", fallback=None)

    if config.has_section("GENETIC"):
        cfg["mutation_rate"]  = config.getfloat("GENETIC", "mutation_rate",  fallback=None)
        cfg["crossover_rate"] = config.getfloat("GENETIC", "crossover_rate", fallback=None)

    if config.has_section("EXECUTION"):
        cfg["generations"] = config.getint("EXECUTION", "generations", fallback=None)
        seed_val = config.get("EXECUTION", "seed", fallback=None)
        cfg["seed"] = int(seed_val) if seed_val and seed_val.strip() != "" else None

    if config.has_section("RESTRICTIONS"):
        # Read 20 per-AA booleans, then merge linked pairs to get 18 effective genes
        restrictions_20 = [
            config.getboolean("RESTRICTIONS", aa.code_3, fallback=True)
            for aa in AMINO_ACIDS
        ]
        cfg["restrictions"] = merge_restrictions_to_18(restrictions_20)
    else:
        cfg["restrictions"] = [True] * N_EFFECTIVE_AA   # default: all modifiable

    if config.has_section("FITNESS"):
        cfg["q_max"]           = config.getfloat("FITNESS", "q_max",           fallback=None)
        cfg["ratio_threshold"] = config.getfloat("FITNESS", "ratio_threshold",  fallback=None)

    # [D2O] section — optional fixed D2O list
    raw_d2o = config.get("D2O", "d2o", fallback="None").strip()
    cfg["d2o"] = None if raw_d2o.lower() == "none" else [int(x) for x in raw_d2o.split()]

    return cfg


def merge_config(cli_args: argparse.Namespace,
                 ini_cfg: Optional[Dict] = None) -> Dict[str, Any]:
    """Merge CLI arguments with INI configuration (CLI takes precedence)."""
    ini_cfg = ini_cfg or {}

    def pick(cli_val, ini_val, default):
        return cli_val if cli_val is not None else (ini_val if ini_val is not None else default)

    return {
        "output_dir":         pick(cli_args.output_dir,  ini_cfg.get("output_dir"),  None),
        "batch_script":       pick(cli_args.batch_script, ini_cfg.get("batch_script"), "./new_parrallel_process_pdb.sh"),
        "population_size":    pick(cli_args.population_size,    ini_cfg.get("population_size"),   6),
        "elitism":            pick(cli_args.elitism,            ini_cfg.get("elitism"),           2),
        "d2o_variation_rate": pick(cli_args.d2o_variation_rate, ini_cfg.get("d2o_variation_rate"),5),
        "generations":        pick(cli_args.generations, ini_cfg.get("generations"), 1),
        "seed":               pick(cli_args.seed,         ini_cfg.get("seed"),         1),
        "q_max":              pick(cli_args.q_max,            ini_cfg.get("q_max"),            0.3),
        "ratio_threshold":    pick(cli_args.ratio_threshold,  ini_cfg.get("ratio_threshold"),  0.01),
        "restrictions":       ini_cfg.get("restrictions", [True] * N_EFFECTIVE_AA),
        "d2o":                pick(getattr(cli_args, 'd2o', None), ini_cfg.get("d2o"), None),
        # Reference options
        "no_default_ref":     cli_args.no_default_ref,
        "ref":                cli_args.ref,
        # Metadata
        "config_file":        cli_args.config,
        "verbose":            cli_args.verbose,
        "pdb_file":           cli_args.pdb_file,
    }


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate all configuration parameters. Raises ValueError on bad values."""
    if cfg["population_size"] <= 0:
        raise ValueError("population_size must be > 0")
    if cfg["population_size"] % 3 != 0:
        raise ValueError(f"population_size must be a multiple of 3 (got {cfg['population_size']})")
    if cfg["elitism"] < 0:
        raise ValueError("elitism must be >= 0")
    if cfg["elitism"] > cfg["population_size"] // 3:
        raise ValueError(f"elitism must be ≤ population_size/3 (max {cfg['population_size'] // 3})")
    # d2o_variation_rate is only meaningful when no fixed d2o list is given
    if cfg.get("d2o") is None and not (0 <= cfg["d2o_variation_rate"] <= 100):
        raise ValueError("d2o_variation_rate must be in [0, 100]")
    if cfg["generations"] < 1:
        raise ValueError("generations must be >= 1")

    # Accept 18 (effective) or 20 (canonical) restriction lengths; auto-convert 20->18
    restr_len = len(cfg["restrictions"])
    if restr_len == len(AMINO_ACIDS):
        cfg["restrictions"] = merge_restrictions_to_18(cfg["restrictions"])
    elif restr_len != N_EFFECTIVE_AA:
        raise ValueError(
            f"restrictions length ({restr_len}) must be "
            f"{N_EFFECTIVE_AA} (effective) or {len(AMINO_ACIDS)} (canonical)"
        )

    if cfg["q_max"] <= 0:
        raise ValueError("q_max must be positive")
    if not (0.0 <= cfg["ratio_threshold"] <= 1.0):
        raise ValueError("ratio_threshold must be in [0.0, 1.0]")
    if cfg["seed"] is not None and cfg["seed"] < 0:
        raise ValueError("seed must be non-negative")
    # Validate fixed d2o list if provided
    d2o = cfg.get("d2o")
    if d2o is not None:
        if not isinstance(d2o, list) or len(d2o) == 0:
            raise ValueError("d2o must be a non-empty list of integers")
        for v in d2o:
            if not isinstance(v, int) or not (0 <= v <= 100):
                raise ValueError(f"d2o value {v!r} must be an integer in [0, 100]")
    # Warn if no refs will be created
    if cfg.get("no_default_ref") and not cfg.get("ref"):
        logger.warning(
            "--no_default_ref is set with no --ref provided. "
            "Ensure ref/ already contains .dat reference files."
        )
    # Validate user-provided ref paths
    if cfg.get("ref"):
        for p in cfg["ref"]:
            if not Path(p).exists():
                raise ValueError(f"Reference PDB not found: {p}")


# ============================================================================
#                       OUTPUT DIRECTORY
# ============================================================================

def create_output_directory(output_dir: Optional[str], pdb_file: str):
    if output_dir is None:
        output_path = Path(Path(pdb_file).stem + "_deuterated_pdbs")
    else:
        output_path = Path(output_dir)

    output_path.mkdir(exist_ok=True, parents=True)
    ref_path = output_path / "ref"
    ref_path.mkdir(exist_ok=True)
    logger.info(f"PDB output directory : {output_path.absolute()}")
    return output_path, ref_path


# ============================================================================
#                       PROTEIN COMPOSITION ANALYSIS
# ============================================================================

def apply_missing_aa_to_restrictions(restrictions: List[bool],
                                     missing_aa: List[str]) -> List[bool]:
    """
    Return a copy of *restrictions* with the effective genes corresponding to
    absent amino acids forced to False.

    When a linked group (e.g. ASN+ASP) has ALL its members absent, the
    group gene is disabled.  If only one member of a linked pair is absent
    the gene remains as configured in *restrictions* — the present member
    will still be deuterated, which is the conservative choice.

    Args:
        restrictions: 18-element effective-gene restriction list.
        missing_aa:   List of 3-letter codes absent from the protein
                      (from PdbDeuteration.missing_aa).

    Returns:
        New 18-element list with absent-AA genes disabled.
    """
    from __init__ import LINKED_AA_GROUPS  # avoid circular at module level

    updated = list(restrictions)
    missing_set = set(missing_aa)

    for gene_idx, group in enumerate(LINKED_AA_GROUPS):
        # Disable the gene only when every AA in the group is absent
        if all(code in missing_set for code in group):
            if updated[gene_idx]:
                logger.info(
                    f"  Restriction auto-disabled for gene {gene_idx} "
                    f"({'+'.join(group)}): absent from structure"
                )
                updated[gene_idx] = False

    return updated


def generate_protein_plots(pdb_analyzer: PdbDeuteration,
                            plot_dir: Path) -> None:
    """
    Generate and save two barplot PNGs into *plot_dir*:

      aa_count.png       — number of unique residues per AA type
                           (mirrors barplot.py)
      aa_hydrogen_count.png — total H atoms contributed by each AA type
                           (mirrors barplot_H.py)

    Both plots use horizontal bars with the canonical AA order reversed
    (Val at top, Ala at bottom) to match the reference scripts.

    Args:
        pdb_analyzer: A PdbDeuteration instance whose aa_count and
                      aa_hydrogen_count attributes have been populated
                      (i.e. __init__ has completed).
        plot_dir:     Directory where PNG files will be written.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')   # non-interactive backend for file output
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping protein composition plots")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    codes  = [aa.code_3 for aa in AMINO_ACIDS]          # 20 codes, forward order
    labels = _AA_PLOT_LABELS                             # short labels, forward order

    # Reversed order: Val at top, Ala at bottom (horizontal bar convention)
    codes_rev  = codes[::-1]
    labels_rev = labels[::-1]

    #  Plot 1 : AA residue counts                                        
    values_count = [pdb_analyzer.aa_count.get(code, 0) for code in codes_rev]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels_rev, values_count, color='steelblue')
    ax.set_xlabel("Number of residues", fontsize=12)
    ax.set_ylabel("Amino acid", fontsize=12)
    ax.set_title("Residue count per amino acid type", fontsize=13, fontweight='bold')

    # Annotate count at end of each bar (skip zeros)
    for i, v in enumerate(values_count):
        if v > 0:
            ax.text(v + max(values_count) * 0.01, i, str(v), va='center', fontsize=9)

    ax.set_xlim(0, max(values_count) * 1.12 if any(v > 0 for v in values_count) else 1)
    plt.tight_layout()
    out_count = plot_dir / "aa_count.png"
    plt.savefig(out_count, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Plot saved : {out_count.name}")

   
    #  Plot 2 : H atoms per AA type                                      
    values_h = [pdb_analyzer.aa_hydrogen_count.get(code, 0) for code in codes_rev]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels_rev, values_h, color='darkorange')
    ax.set_xlabel("Number of H atoms", fontsize=12)
    ax.set_ylabel("Amino acid", fontsize=12)
    ax.set_title("Total H atoms per amino acid type", fontsize=13, fontweight='bold')

    for i, v in enumerate(values_h):
        if v > 0:
            ax.text(v + max(values_h) * 0.01, i, str(v), va='center', fontsize=9)

    ax.set_xlim(0, max(values_h) * 1.12 if any(v > 0 for v in values_h) else 1)
    plt.tight_layout()
    out_h = plot_dir / "aa_hydrogen_count.png"
    plt.savefig(out_h, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Plot saved : {out_h.name}")


# ============================================================================
#                       PDB GENERATION
# ============================================================================

def create_protonated_reference_pdbs(pdb_file: str,
                                      ref_dir: Path,
                                      restrictions: List[bool]) -> None:
    """
    Create the two default reference PDB files:
      - protonated protein in D2O  (all labile H exchanged via d2o=100)
      - protonated protein in H2O  (no deuteration, d2o=0)
    """
    logger.info(">>> Creating default reference PDBs (protonated in D2O / H2O)")
    ref_deut_vector = [False] * len(AMINO_ACIDS)   # 20-element, no AA deuteration

    # Protonated in D2O (all labile H exchanged)
    deut_pdb  = PdbDeuteration(pdb_file)
    deut_pdb.apply_deuteration(ref_deut_vector, d2o_percent=100)
    deut_path = ref_dir / f"{Path(pdb_file).stem}_total_deuteration.pdb"
    deut_pdb.save(str(deut_path))
    logger.info(f"  Deuteration reference : {deut_path.name}")

    # Protonated in H2O (all H, no exchange)
    prot_pdb  = PdbDeuteration(pdb_file)
    prot_pdb.apply_deuteration(ref_deut_vector, d2o_percent=0)
    prot_path = ref_dir / f"{Path(pdb_file).stem}_total_protonation.pdb"
    prot_pdb.save(str(prot_path))
    logger.info(f"  Protonation reference : {prot_path.name}")


def copy_user_reference_pdbs(ref_paths: List[str], ref_dir: Path) -> None:
    """Copy user-provided reference PDB files into ref_dir."""
    logger.info(f">>> Copying {len(ref_paths)} user-provided reference PDB(s) to ref/")
    for src in ref_paths:
        src_path = Path(src)
        if not src_path.exists():
            logger.warning(f"  Reference PDB not found, skipping: {src}")
            continue
        dst = ref_dir / src_path.name
        shutil.copy2(src_path, dst)
        logger.info(f"  Copied: {src_path.name}")


def generate_pdbs_for_chromosomes(pdb_file: str,
                                   chromosomes: List[Chromosome],
                                   output_dir: Path) -> List[str]:
    """
    Generate one deuterated PDB file per chromosome.

    Each chromosome carries an 18-element deuteration vector.  Before passing
    it to PdbDeuteration, we expand it to the 20-element canonical vector so
    that both amino acids in each linked pair (ASN+ASP, GLU+GLN) are
    deuterated when their shared gene is True.
    """
    logger.info(f"Generating {len(chromosomes)} deuterated PDB file(s)…")
    generated = []

    for chrom in chromosomes:
        filename = get_pdb_filename(chrom)
        out_path = output_dir / filename

        deut_vector_20 = expand_deuteration_vector(chrom.deuteration)

        try:
            deuterator = PdbDeuteration(pdb_file)
            deuterator.apply_deuteration(deut_vector_20, chrom.d2o)     # 20-element
            chrom.H           = deuterator.stats['hydrogen_atoms']
            chrom.D           = deuterator.stats['deuterium_atoms']
            chrom.non_labile_D = deuterator.stats['non_labile_D']
            deuterator.save(str(out_path))
            generated.append(str(out_path))
            logger.debug(f"  Written: {filename}")
        except Exception as exc:
            logger.error(f"  FAILED {filename}: {exc}")

    logger.info(f"  {len(generated)} PDB file(s) written.")
    return generated


# ============================================================================
#                       SANS SIMULATION
# ============================================================================

def run_batch_processing(output_dir: Path,
                          batch_script: str,
                          new_pdb_files: Optional[List[str]] = None) -> str:
    """
    Execute the external batch script for Pepsi-SANS simulation.

    Args:
        output_dir:    Directory containing PDB files and ref/ subfolder.
        batch_script:  Path to the shell script.
        new_pdb_files: If given, only these files are simulated (incremental).
                       If None, all PDB files in output_dir are processed.

    Returns:
        Absolute path (string) to the primus_out directory.
    """
    logger.info("=" * 70)
    logger.info(">>> Running Pepsi-SANS simulation…")
    logger.info("=" * 70)

    tmp_list_file: Optional[str] = None
    try:
        if new_pdb_files:
            fd, tmp_list_file = tempfile.mkstemp(suffix=".txt", prefix="pdb_list_")
            with os.fdopen(fd, 'w') as fh:
                for p in new_pdb_files:
                    fh.write(p + "\n")
            cmd = [batch_script, str(output_dir), tmp_list_file]
            logger.info(f"  Incremental mode: {len(new_pdb_files)} new file(s)")
        else:
            cmd = [batch_script, str(output_dir)]
            logger.info("  Full mode: all PDB files in output directory")

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(result.stdout)

        folder_name = output_dir.name
        prefix      = folder_name.replace("_deuterated_pdbs", "")
        primus_dir  = str(output_dir.parent / f"{prefix}_primus_out")
        logger.info(f"  SANS data directory  : {primus_dir}")
        return primus_dir

    except subprocess.CalledProcessError as exc:
        logger.error(f"Batch processing failed (exit code {exc.returncode})")
        logger.error(exc.stderr)
        raise RuntimeError(f"Batch script error: {exc}") from exc
    except FileNotFoundError:
        raise RuntimeError(f"Batch script not found: {batch_script}")
    finally:
        if tmp_list_file and os.path.exists(tmp_list_file):
            os.unlink(tmp_list_file)


# ============================================================================
#                       FITNESS EVALUATION
# ============================================================================

def evaluate_fitness(primus_dir: str,
                     population: List[Chromosome],
                     q_max: float,
                     ratio_threshold: float,
                     deut_ref: Optional[str] = None,
                     prot_ref: Optional[str] = None) -> None:
    """
    Evaluate fitness for all chromosomes by matching .dat filenames.

    Calls evaluate_population_fitness() which returns raw fitness scores for
    every .dat file in primus_dir (alphabetical order).  Each score is then
    mapped back to the corresponding chromosome via the chromosome's expected
    filename stem.

    Args:
        primus_dir:       Path to the Pepsi-SANS output directory.
        population:       Full current population (tier-1 + tier-2 + tier-3).
        q_max:            q truncation limit.
        ratio_threshold:  Minimum Imax/background ratio threshold.
        deut_ref:         Exact deuterated reference filename (or None for auto-detect).
        prot_ref:         Exact protonated reference filename (or None for auto-detect).
    """
    logger.info("=" * 70)
    logger.info(">>> Evaluating fitness from SANS data…")
    logger.info("=" * 70)

    chrom_by_stem: Dict[str, Chromosome] = {
        Path(get_pdb_filename(c)).stem: c for c in population
    }

    try:
        fitness_scores, dat_files, ratios = evaluate_population_fitness(
            directory=primus_dir,
            q_max=q_max,
            ratio_threshold=ratio_threshold,
        )
    except Exception as exc:
        logger.error(f"Fitness evaluation failed: {exc}")
        raise RuntimeError(f"Fitness evaluation error: {exc}") from exc

    matched, unmatched_files = 0, []
    for score, file_path, ratio in zip(fitness_scores, dat_files, ratios):
        stem = Path(file_path).stem
        if stem in chrom_by_stem:
            chrom_by_stem[stem].fitness = float(score)
            chrom_by_stem[stem].ratio = float(ratio)
            matched += 1
        else:
            unmatched_files.append(Path(file_path).name)

    if unmatched_files:
        logger.warning(
            f"  {len(unmatched_files)} .dat file(s) had no matching chromosome: "
            + ", ".join(unmatched_files[:5])
            + (" …" if len(unmatched_files) > 5 else "")
        )

    passed = int(np.sum(fitness_scores > 0))
    logger.info(f"  Matched {matched}/{len(population)} chromosome(s). Passed ratio check: {passed}/{len(fitness_scores)}")


# ============================================================================
#                       FILE CLEANUP
# ============================================================================

def cleanup_non_tier1_files(output_dir: Path,
                             primus_dir: Optional[str],
                             tier1_population: List[Chromosome]) -> None:
    """
    Remove all PDB and SANS output files that do NOT belong to tier-1 chromosomes.

    Tier-1 chromosomes retain their original creation filenames so their files
    are easily identified by stem.  The ref/ subfolder is never touched.
    """
    tier1_stems: Set[str] = {Path(get_pdb_filename(c)).stem for c in tier1_population}

    # PDB files
    removed_pdb = 0
    for pdb in list(output_dir.glob("gen*_Chr*.pdb")):
        if pdb.stem not in tier1_stems:
            try:
                pdb.unlink()
                removed_pdb += 1
                logger.debug(f"  Removed PDB: {pdb.name}")
            except OSError as exc:
                logger.warning(f"  Could not remove {pdb.name}: {exc}")

    # SANS output files
    removed_sans = 0
    if primus_dir:
        primus_path = Path(primus_dir)
        for ext in ('.dat', '.out', '.log'):
            for f in list(primus_path.glob(f"gen*_Chr*{ext}")):
                if f.stem not in tier1_stems:
                    try:
                        f.unlink()
                        removed_sans += 1
                        logger.debug(f"  Removed SANS: {f.name}")
                    except OSError as exc:
                        logger.warning(f"  Could not remove {f.name}: {exc}")

    logger.info(
        f"  Cleanup: removed {removed_pdb} PDB, {removed_sans} SANS file(s). "
        f"Kept {len(tier1_stems)} tier-1 file(s)."
    )


# ============================================================================
#                       DISPLAY & SUMMARY
# ============================================================================

def display_population_summary(population: List[Chromosome],
                                sorted_indices: List[int],
                                generation: int) -> None:
    """Log a short summary of the top-3 chromosomes using their original filenames."""
    top_n = min(3, len(sorted_indices))
    logger.info(f"\n>>> Generation {generation} — top {top_n} chromosome(s):")
    for rank in range(top_n):
        chrom = population[sorted_indices[rank]]
        logger.info(
            f"  {rank + 1:2d}. {get_pdb_filename(chrom)}"
            f"  fitness={chrom.fitness:.6f}"
            f"  H={chrom.H}  D={chrom.D}"
            f"  %D={(chrom.D/(chrom.H + chrom.D))*100:.2f}"
            f"  ratio={chrom.ratio:.3f}"
        )


def save_population_summary(population: List[Chromosome],
                             sorted_indices: List[int],
                             output_dir: Path,
                             generation: int) -> None:
    """
    Save a detailed ranked summary of the population to a text file.

    The 'AA' column now shows the number of deuterated *effective genes*
    (0-18) rather than canonical AAs (0-20), because the chromosome operates
    on the 18-gene linked-pair space.
    """
    summary_file  = output_dir / f"generation_{generation:02d}_summary.txt"
    fitness_values = [population[i].fitness for i in sorted_indices
                      if population[i].fitness is not None]

    with open(summary_file, 'w', encoding='utf-8') as fh:
        fh.write("=" * 150 + "\n")
        fh.write(f"POPULATION SUMMARY - GENERATION {generation}\n")
        fh.write("=" * 150 + "\n\n")
        fh.write(f"Population size : {len(population)}\n")
        if fitness_values:
            fh.write(f"Best fitness    : {max(fitness_values):.6f}\n")
            fh.write(f"Average fitness : {float(np.mean(fitness_values)):.6f}\n")
            fh.write(f"Worst fitness   : {min(fitness_values):.6f}\n")
        fh.write("\n")
        fh.write(
            f"{'Rank':<6} {'PDB filename (creation name)':<55} "
            f"{'D2O%':<6} {'AA':<5} {'ratio':<14} {'Fitness':<14} "
            f"{'D%':<14} {'Non_labile_D%'} {'Created':<12}\n"
        )
        fh.write("-" * 150 + "\n")
        for rank, idx in enumerate(sorted_indices, 1):
            chrom = population[idx]
            filename = get_pdb_filename(chrom)
            fh.write(
                f"{rank:<6} {filename:<55} {chrom.d2o:<6} "
                f"{sum(chrom.deuteration):<5} {chrom.ratio:<14.3f} {chrom.fitness:<14.6f} "
                f"{(chrom.D/(chrom.H + chrom.D))*100:<14.2f} "
                f"{(chrom.non_labile_D / (chrom.H + chrom.D))*100:<14.2f}"
                f"gen{chrom.generation:02d}_idx{chrom.index:03d}\n"
            )
        fh.write("=" * 150 + "\n")

    logger.info(f"  Summary saved : {summary_file.name}")


def save_best_fitness_summary(best_chrom: Chromosome,
                               generation: int,
                               summary_path: Path) -> None:
    """
    Append one row per generation to the best-fitness CSV.
    """
    deut_aas = [
        EFFECTIVE_AMINO_ACIDS[i].code_3
        for i, d in enumerate(best_chrom.deuteration)
        if d
    ]
    deut_str = ";".join(deut_aas) if deut_aas else "none"
    filename = get_pdb_filename(best_chrom)
    d_str = (best_chrom.D / (best_chrom.H + best_chrom.D)) * 100
    non_labile_d_str = (best_chrom.non_labile_D / (best_chrom.H + best_chrom.D)) * 100

    write_header = not summary_path.exists()
    with open(summary_path, 'a', encoding='utf-8') as fh:
        if write_header:
            fh.write(
                "generation,fitness,d2o_percent,n_deuterated_aa,ratio,%D,%Non_labile_D%,"
                "deuterated_aa_list,pdb_filename,created_generation,created_index\n"
            )
        fh.write(
            f"{generation},{best_chrom.fitness:.8f},{best_chrom.d2o},"
            f"{sum(best_chrom.deuteration)},{best_chrom.ratio:.3f},"
            f"{d_str:.2f},{non_labile_d_str:.2f},"
            f"{deut_str},{filename},{best_chrom.generation},{best_chrom.index}\n"
        )
    logger.info(
        f"  Best fitness CSV: gen={generation}, "
        f"fitness={best_chrom.fitness:.6f}, D2O={best_chrom.d2o}%"
    )


# ============================================================================
#                       ELITISM GUARD
# ============================================================================

def check_fitness_non_decreasing(population, sorted_indices, previous_best, generation):
    current_best = population[sorted_indices[0]].fitness
    if previous_best is None:
        logger.info(f"  Best fitness (gen {generation}): {current_best:.6f}")
    elif current_best < previous_best - 1e-10:
        logger.warning(
            f"  ELITISM VIOLATION at generation {generation}: "
            f"best fitness decreased from {previous_best:.6f} to {current_best:.6f}."
        )
    else:
        logger.info(
            f"  Best fitness (gen {generation}): {current_best:.6f} "
            f"(Δ={current_best - previous_best:+.6f})"
        )
    return current_best


# ============================================================================
#                       FINAL RESULT FOLDER
# ============================================================================

def create_final_result_folder(best_chrom, pdb_stem, output_dir,
                                primus_dir, best_summary_path):
    folder_name  = output_dir.name
    prefix       = folder_name.replace("_deuterated_pdbs", "")
    final_dir    = output_dir.parent / f"{prefix}_final_results"
    primus_path  = Path(primus_dir)

    pdb_dir      = final_dir / "pdb"
    pdb_ref_dir  = pdb_dir / "ref"
    sans_dir     = final_dir / "sans_simulation"
    sans_ref_dir = sans_dir / "ref"

    for d in (final_dir, pdb_dir, pdb_ref_dir, sans_dir, sans_ref_dir):
        d.mkdir(parents=True, exist_ok=True)

    base_name = f"{pdb_stem}_best_fitness"
    logger.info("\n" + "=" * 70)
    logger.info(">>> Assembling final result folder")
    logger.info(f"  Destination : {final_dir.absolute()}")

    # Single-row CSV (header + best row)
    csv_dest = final_dir / f"{base_name}.csv"
    with open(best_summary_path, "r", encoding="utf-8") as src_fh:
        lines = [l for l in src_fh.readlines() if l.strip()]
    with open(csv_dest, "w", encoding="utf-8") as dst_fh:
        dst_fh.write(lines[0])
        dst_fh.write(lines[-1])
    logger.info(f"  CSV written             : {csv_dest.name}")

    # Best PDB
    best_pdb_src = output_dir / get_pdb_filename(best_chrom)
    if best_pdb_src.exists():
        shutil.copy2(best_pdb_src, pdb_dir / f"{base_name}.pdb")
        logger.info("  Best PDB copied")
    else:
        logger.warning(f"  Best PDB NOT found: {best_pdb_src.name}")

    # Reference PDBs
    for src in sorted((output_dir / "ref").glob("*.pdb")):
        shutil.copy2(src, pdb_ref_dir / src.name)
        logger.info(f"  Ref PDB copied: {src.name}")

    # Best SANS outputs
    best_dat_stem = Path(get_pdb_filename(best_chrom)).stem
    for ext in (".dat", ".log", ".out"):
        src = primus_path / f"{best_dat_stem}{ext}"
        if src.exists():
            shutil.copy2(src, sans_dir / f"{base_name}{ext}")

    # Reference SANS files
    ref_sans_dir = primus_path / "ref"
    if ref_sans_dir.is_dir():
        for dat_src in sorted(ref_sans_dir.glob("*.dat")):
            shutil.copy2(dat_src, sans_ref_dir / dat_src.name)
            for ext in (".log", ".out"):
                s = dat_src.with_suffix(ext)
                if s.exists():
                    shutil.copy2(s, sans_ref_dir / s.name)

    logger.info(f"  Final result folder ready: {final_dir.absolute()}")
    logger.info("=" * 70)
    return final_dir


# ============================================================================
#                           MAIN EXECUTION
# ============================================================================

def main():
    cli_args = parse_arguments()

    # ---------- Load & merge configuration ----------
    ini_cfg = None
    if cli_args.config:
        try:
            ini_cfg = load_config_ini(cli_args.config)
            logger.info(f"Loaded configuration from: {cli_args.config}")
        except (FileNotFoundError, ValueError) as exc:
            logger.error(f"Config error: {exc}")
            sys.exit(1)

    try:
        cfg = merge_config(cli_args, ini_cfg)
        validate_config(cfg)
    except ValueError as exc:
        logger.error(f"Configuration validation failed: {exc}")
        sys.exit(1)

    # ---------- Random seed ----------
    if cfg["seed"] is not None:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        logger.info(f"Random seed set to: {cfg['seed']}")

    # ---------- Validate input files ----------
    pdb_path = Path(cfg['pdb_file'])
    if not pdb_path.exists():
        logger.error(f"PDB file not found: {pdb_path}")
        sys.exit(1)
    if not Path(cfg['batch_script']).exists():
        logger.error(f"Batch script not found: {cfg['batch_script']}")
        sys.exit(1)

    # ---------- Analyse PDB composition BEFORE creating population ----------
    logger.info("=" * 70)
    logger.info(">>> Analysing protein composition")
    logger.info("=" * 70)
    try:
        pdb_analyzer = PdbDeuteration(str(pdb_path))
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error(f"PDB analysis failed: {exc}")
        sys.exit(1)

    total_residues = sum(pdb_analyzer.aa_count.values())
    total_h        = sum(pdb_analyzer.aa_hydrogen_count.values())
    logger.info(f"  Total residues  : {total_residues}")
    logger.info(f"  Total H atoms   : {total_h}")

    if pdb_analyzer.missing_aa:
        logger.info(
            f"  Absent AAs ({len(pdb_analyzer.missing_aa)}): "
            + ", ".join(pdb_analyzer.missing_aa)
        )
    else:
        logger.info("  All 20 standard amino acid types are present")

    # Update restrictions: disable effective genes for fully absent AAs
    cfg['restrictions'] = apply_missing_aa_to_restrictions(
        cfg['restrictions'], pdb_analyzer.missing_aa
    )

    # ---------- Display configuration ----------
    logger.info("=" * 70)
    logger.info("                SANS DEUTERATION OPTIMISATION")
    logger.info("=" * 70)
    logger.info(f"Source PDB           : {cfg['pdb_file']}")
    logger.info(f"Population size      : {cfg['population_size']}")
    logger.info(f"Elitism              : {cfg['elitism']}")
    if cfg.get('d2o') is None:
        logger.info(f"D2O variation rate   : ±{cfg['d2o_variation_rate']}%")
    else:
        logger.info(f"D2O fixed values     : {cfg['d2o']}")
    logger.info(f"Generations          : {cfg['generations']}")
    logger.info(f"Q max (fitness)      : {cfg['q_max']} Å⁻¹")
    logger.info(f"Ratio threshold      : {cfg['ratio_threshold']}")
    logger.info(f"Effective gene count : {N_EFFECTIVE_AA}  "
                f"(ASN+ASP linked, GLU+GLN linked)")
    logger.info(f"Default references   : "
                f"{'no' if cfg.get('no_default_ref') else 'yes (protonated in D2O / H2O)'}")
    if cfg.get("ref"):
        logger.info(f"Extra references     : {cfg['ref']}")
    logger.info("=" * 70)

    # ---------- Create output directories ----------
    output_dir, ref_dir = create_output_directory(cfg['output_dir'], str(pdb_path))
    best_summary_path = output_dir / "best_fitness_summary.csv"

    # ---------- Generate protein composition plots ----------
    plot_dir = output_dir / "plot"
    logger.info(">>> Generating protein composition plots")
    generate_protein_plots(pdb_analyzer, plot_dir)

    # Create default references unless --no_default_ref
    if not cfg.get("no_default_ref"):
        create_protonated_reference_pdbs(str(pdb_path), ref_dir, cfg['restrictions'])

    # Copy user-provided reference PDBs
    if cfg.get("ref"):
        copy_user_reference_pdbs(cfg["ref"], ref_dir)

    # ---------- Initialise population generator ----------
    generator = PopulationGenerator(
        aa_list=EFFECTIVE_AMINO_ACIDS,        # 18 effective genes
        modifiable=cfg['restrictions'],        # 18-element list (missing AAs already disabled)
        population_size=cfg['population_size'],
        elitism=cfg['elitism'],
        d2o_variation_rate=cfg['d2o_variation_rate'],
        d2o=cfg['d2o'],
    )
    tier_size = cfg['population_size'] // 3

    # ---- GENERATION 0 ----
    logger.info("\n" + "=" * 70)
    logger.info(">>> GENERATION 0 — creating initial population")
    logger.info("=" * 70)

    population = generator.generate_initial_population(generation=0)
    generate_pdbs_for_chromosomes(str(pdb_path), population, output_dir)
    primus_dir = run_batch_processing(output_dir, cfg['batch_script'])
    evaluate_fitness(primus_dir, population, cfg['q_max'], cfg['ratio_threshold'])

    sorted_indices = get_sorted_indices(population)
    previous_best_fitness = check_fitness_non_decreasing(population, sorted_indices, None, 0)
    display_population_summary(population, sorted_indices, 0)
    save_population_summary(population, sorted_indices, output_dir, 0)
    save_best_fitness_summary(population[sorted_indices[0]], 0, best_summary_path)

    # ---- SUBSEQUENT GENERATIONS ----
    for gen in range(1, cfg['generations']):
        logger.info("\n" + "=" * 70)
        logger.info(f">>> GENERATION {gen} — population evolution")
        logger.info("=" * 70)

        new_population = generator.generate_next_generation(
            previous_population=population,
            d2o_variation_rate=cfg['d2o_variation_rate'],
            new_generation=gen,
        )

        tier1 = new_population[:tier_size]
        tier2_and_3 = new_population[tier_size:]

        cleanup_non_tier1_files(output_dir, primus_dir, tier1)

        new_pdb_files = generate_pdbs_for_chromosomes(str(pdb_path), tier2_and_3, output_dir)
        primus_dir    = run_batch_processing(output_dir, cfg['batch_script'],
                                             new_pdb_files=new_pdb_files)

        evaluate_fitness(primus_dir, new_population, cfg['q_max'], cfg['ratio_threshold'])

        sorted_indices        = get_sorted_indices(new_population)
        previous_best_fitness = check_fitness_non_decreasing(
            new_population, sorted_indices, previous_best_fitness, gen
        )
        display_population_summary(new_population, sorted_indices, gen)
        save_population_summary(new_population, sorted_indices, output_dir, gen)
        save_best_fitness_summary(new_population[sorted_indices[0]], gen, best_summary_path)

        population = new_population

    # ---- FINAL SUMMARY ----
    logger.info("\n" + "=" * 70)
    logger.info("GENETIC ALGORITHM COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    best = population[sorted_indices[0]]
    logger.info("Best solution:")
    logger.info(f"  File    : {get_pdb_filename(best)}")
    logger.info(f"  D2O     : {best.d2o}%")
    # Report effective genes deuterated (max 18) + canonical AAs affected (max 20)
    deut_genes   = sum(best.deuteration)
    deut_vec_20  = expand_deuteration_vector(best.deuteration)
    deut_aa_count = sum(deut_vec_20)
    logger.info(f"  Deut. genes : {deut_genes}/18 effective genes  "
                f"→ {deut_aa_count}/20 canonical AAs deuterated")
    logger.info(f"  %D      : {(best.D / (best.H + best.D))*100:.2f}%")
    logger.info(f"  Fitness : {best.fitness:.6f}")

    create_final_result_folder(
        best_chrom=best,
        pdb_stem=pdb_path.stem,
        output_dir=output_dir,
        primus_dir=primus_dir,
        best_summary_path=best_summary_path,
    )


if __name__ == "__main__":
    try:
        start = time.perf_counter()
        main()
        elapsed = time.perf_counter() - start
        print(f"\nTotal execution time: {elapsed:.2f} s")
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"Fatal error: {exc}")
        sys.exit(1)
