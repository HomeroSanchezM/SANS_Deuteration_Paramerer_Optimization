#!/usr/bin/env python3
"""
Fitness Evaluation for SANS Deuteration Profiles
================================================

This module evaluates the fitness of simulated SANS curves by comparing them
to reference deuterated and protonated experimental curves.

Directory structure expected:
    /path/to/directory/
        ├── file1.dat                # Simulated intensity curves
        ├── file2.dat
        ├── ...
        └── ref/                    # Reference data subfolder
            ├── *_deuterated.dat    # Deuterated reference (exact name can be specified)
            └── *_protonated.dat    # Protonated reference

Fitness is defined as the sum of the normalized areas between:
    1. The scaled simulated curve and the deuterated reference
    2. The scaled simulated curve and the protonated reference

A higher fitness score (closer to 1) indicates a better match.
"""

import numpy as np
import argparse
import os
import glob
import logging
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.integrate import simpson

# ============================================================================
#                           LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
#                           ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """
    Parses command-line arguments for fitness evaluation.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate fitness of simulated SANS curves against reference data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Directory structure:
  <directory>/
    ├── *.dat                    # Simulated curves to evaluate
    └── ref/                     # Reference data (mandatory subfolder)
        ├── *deuterated.dat      # Deuterated reference
        └── *protonated.dat      # Protonated reference

Examples:
  # Basic usage (assumes default naming in ref/)
  python fitness_evaluation.py /path/to/simulations

  # Specify custom reference filenames
  python fitness_evaluation.py /path/to/simulations --deut-ref D2O.dat --prot-ref H2O.dat

  # Adjust q-range and I0 threshold
  python fitness_evaluation.py /path/to/simulations --q-max 0.35 --i0-threshold 0.15

  # Quiet mode (minimal output)
  python fitness_evaluation.py /path/to/simulations --quiet

  # Verbose mode (debug information)
  python fitness_evaluation.py /path/to/simulations --verbose
        """
    )

    # ==================== POSITIONAL ARGUMENTS ====================
    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing simulated .dat files and a "ref/" subfolder'
    )

    # ==================== REFERENCE FILE OPTIONS ====================
    ref_group = parser.add_argument_group('Reference files')
    ref_group.add_argument(
        '--deut-ref',
        type=str,
        default=None,
        help='Filename of deuterated reference inside ref/ (default: auto-detect with "*deuterated.dat")'
    )
    ref_group.add_argument(
        '--prot-ref',
        type=str,
        default=None,
        help='Filename of protonated reference inside ref/ (default: auto-detect with "*protonated.dat")'
    )

    # ==================== EVALUATION PARAMETERS ====================
    param_group = parser.add_argument_group('Evaluation parameters')
    param_group.add_argument(
        '--q-max',
        type=float,
        default=0.3,
        help='Maximum q value for truncation (default: 0.3 Å⁻¹)'
    )
    param_group.add_argument(
        '--i0-threshold',
        type=float,
        default=0.2,
        help='Minimum I(0) as fraction of protonated I(0) to pass filter (default: 0.2)'
    )

    # ==================== OUTPUT CONTROL ====================
    output_group = parser.add_argument_group('Output control')
    output_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose debug output'
    )
    output_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress all non‑essential output (only print final fitness array)'
    )

    args = parser.parse_args()

    # Adjust logging level
    if args.quiet:
        logger.setLevel(logging.WARNING)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return args


# ============================================================================
#                           FILE DISCOVERY
# ============================================================================

def find_dat_files(directory):
    """
    Find all .dat files in the specified directory.

    Args:
        directory (str): Path to the directory

    Returns:
        List[str]: Sorted list of paths to .dat files

    Raises:
        NotADirectoryError: If the path is not a directory
        FileNotFoundError: If no .dat files are found
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"'{directory}' is not a valid directory")

    # Search for .dat files
    dat_files = glob.glob(os.path.join(directory, "*.dat"))
    dat_files = [f for f in dat_files if os.path.isfile(f)]

    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in directory '{directory}'")

    return sorted(dat_files)


def find_reference_files(ref_dir, deut_ref=None, prot_ref=None):
    """
    Locate deuterated and protonated reference files in ref/ subdirectory.

    Args:
        ref_dir (str): Path to the ref/ directory
        deut_ref (str, optional): Exact filename for deuterated reference
        prot_ref (str, optional): Exact filename for protonated reference

    Returns:
        tuple: (deut_path, prot_path)

    Raises:
        FileNotFoundError: If reference files cannot be located
    """
    if not os.path.isdir(ref_dir):
        raise NotADirectoryError(f"Reference directory not found: {ref_dir}")

    # Case 1: Exact filenames provided
    if deut_ref:
        deut_path = os.path.join(ref_dir, deut_ref)
        if not os.path.isfile(deut_path):
            raise FileNotFoundError(f"Deuteration reference not found: {deut_path}")
    else:
        # Auto-detect: look for '*deuteration.dat' (case insensitive)
        deut_candidates = glob.glob(os.path.join(ref_dir, "*deuteration.dat")) + \
                          glob.glob(os.path.join(ref_dir, "*deuteration.DAT"))
        if not deut_candidates:
            raise FileNotFoundError(
                "No deuterated reference found. Provide --deut-ref or ensure a file matching '*deuteration.dat' exists in ref/."
            )
        deut_path = sorted(deut_candidates)[0]
        logger.info(f"Auto-detected deuterated reference: {os.path.basename(deut_path)}")

    if prot_ref:
        prot_path = os.path.join(ref_dir, prot_ref)
        if not os.path.isfile(prot_path):
            raise FileNotFoundError(f"Protonation reference not found: {prot_path}")
    else:
        prot_candidates = glob.glob(os.path.join(ref_dir, "*protonation.dat")) + \
                          glob.glob(os.path.join(ref_dir, "*protonation.DAT"))
        if not prot_candidates:
            raise FileNotFoundError(
                "No protonated reference found. Provide --prot-ref or ensure a file matching '*protonation.dat' exists in ref/."
            )
        prot_path = sorted(prot_candidates)[0]
        logger.info(f"Auto-detected protonated reference: {os.path.basename(prot_path)}")

    return deut_path, prot_path


# ============================================================================
#                           PARSING & UTILITIES
# ============================================================================

def parse_sans_file(filepath):
    """
    Parse a SANS .dat file and extract q and I columns.

    Args:
        filepath (str): Path to the .dat file

    Returns:
        tuple: (q, I) numpy arrays
    """
    q_values = []
    I_values = []

    with open(filepath, 'r') as f:
        for line in f:
            # Ignore comment lines
            if line.startswith('#'):
                continue

            # Parse line
            parts = line.split()
            if len(parts) >= 2:
                try:
                    q = float(parts[0])
                    I = float(parts[1])
                    q_values.append(q)
                    I_values.append(I)
                except ValueError:
                    continue

    # Convert to numpy arrays
    q = np.array(q_values)
    I = np.array(I_values)

    if len(q) == 0:
        raise ValueError(f"No valid data points in file: {filepath}")

    return q, I


def truncate_to_q_max(q, I, q_max):
    """
    Truncate arrays to keep only values where q <= q_max.

    Args:
        q (np.array): q values
        I (np.array): I values
        q_max (float): Maximum q value

    Returns:
        tuple: (q_truncated, I_truncated)
    """
    mask = q <= q_max
    return q[mask], I[mask]


def regrid_curve(q_source, I_source, q_target):
    """
    Interpolate a curve onto a target q grid.

    Args:
        q_source (np.array): Original q values
        I_source (np.array): Original I values
        q_target (np.array): Target q values

    Returns:
        np.array: Interpolated I values
    """
    # Use cubic interpolation to regrid
    I_regridded = griddata(q_source, I_source, q_target, method='cubic')

    # Fallback to linear interpolation for NaN values
    if np.any(np.isnan(I_regridded)):
        # Fall back to linear interpolation for NaN values
        I_linear = griddata(q_source, I_source, q_target, method='linear')
        I_regridded = np.where(np.isnan(I_regridded), I_linear, I_regridded)
    
    return I_regridded


def scale_curves(q, I_reference, I_to_scale):
    """
    Scale I_to_scale to match I_reference using linear model: I_scaled = a * I_to_scale + b.

    Args:
        q (np.array): q values (same for both curves)
        I_reference (np.array): Reference intensity
        I_to_scale (np.array): Intensity to scale

    Returns:
        tuple: (I_scaled, a, b, chi2)
    """
    def linear_model(I, a, b):
        return a * I + b

    try:
        params, _ = curve_fit(linear_model, I_to_scale, I_reference)
        a, b = params
        I_scaled = a * I_to_scale + b
        
        # Calculate chi^2 (area difference metric)
        chi2 = np.sum((I_reference - I_scaled) ** 2)
        
        return I_scaled, a, b, chi2
    
    except Exception as e:
        logger.warning(f"Scaling failed: {e}")
        return I_to_scale, 1.0, 0.0, np.inf


def calculate_area_difference(q, I1, I2):
    """
    Compute the area between two curves using Simpson integration.

    Args:
        q (np.array): q values
        I1 (np.array): First intensity curve
        I2 (np.array): Second intensity curve
    
    Returns:
        float: Area between the two curves
    """
    # Calculate absolute difference
    diff = np.abs(I1 - I2)
    
    # Integrate using Simpson's rule
    area = simpson(diff, x=q)
    
    return area


def i0_threshold_check(I_prot, I, threshold_ratio):
    """
    Check if I[0] >= threshold_ratio * I_prot[0].

    Args:
        I_prot (np.array): Protonated reference intensity
        I (np.array): Simulated intensity
        threshold_ratio (float): Minimum ratio

    Returns:
        bool: True if condition met
    """
    return I[0] >= threshold_ratio * I_prot[0]


def scaling_and_compare(q, I, I_deut, I_prot, q_max):
    """
    Complete scaling and comparison pipeline.

    Args:
        q (np.array): q values for I
        I (np.array): Simulated intensity
        I_deut (np.array): Deuterated reference
        I_prot (np.array): Protonated reference
        q_max (float): Truncation limit

    Returns:
        float: Sum of area differences (deut + prot)
    """
    # 1. Truncate simulated curve
    q_trunc, I_trunc = truncate_to_q_max(q, I, q_max)

    # 2. Truncate references to same q range
    q_deut_trunc, I_deut_trunc = truncate_to_q_max(q, I_deut, q_max)
    q_prot_trunc, I_prot_trunc = truncate_to_q_max(q, I_prot, q_max)

    # 3. Regrid everything onto the simulated q grid (ensures same points)
    #    (We assume the simulated q grid is the finest / most relevant)
    I_deut_regrid = regrid_curve(q_deut_trunc, I_deut_trunc, q_trunc)
    I_prot_regrid = regrid_curve(q_prot_trunc, I_prot_trunc, q_trunc)

    # 4. Scale and compare with deuterated reference
    I_scaled_deut, _, _, _ = scale_curves(q_trunc, I_deut_regrid, I_trunc)
    area_deut = calculate_area_difference(q_trunc, I_scaled_deut, I_deut_regrid)

    # 5. Scale and compare with protonated reference
    I_scaled_prot, _, _, _ = scale_curves(q_trunc, I_prot_regrid, I_trunc)
    area_prot = calculate_area_difference(q_trunc, I_scaled_prot, I_prot_regrid)

    return area_deut + area_prot


def fitness(q, I, I_deut, I_prot, q_max, i0_threshold):
    """
    Compute raw fitness score (area difference) for a single curve.

    Args:
        q (np.array): q values
        I (np.array): Simulated intensity
        I_deut (np.array): Deuterated reference
        I_prot (np.array): Protonated reference
        q_max (float): q truncation limit
        i0_threshold (float): Minimum I(0) ratio

    Returns:
        float: Raw area sum, or 0.0 if I0 threshold fails
    """
    if not i0_threshold_check(I_prot, I, i0_threshold):
        logger.debug(f"I0 threshold failed: I[0]={I[0]:.4e}, required ≥ {i0_threshold*I_prot[0]:.4e}")
        return 0.0

    return scaling_and_compare(q, I, I_deut, I_prot, q_max)


def normalize_fitness(fitness_values):
    """
    Normalize fitness scores to [0, 1] (higher area → higher fitness).

    Args:
        fitness_values (list or np.array): Raw fitness scores

    Returns:
        np.array: Normalized scores
    """
    arr = np.array(fitness_values, dtype=float)

    if len(arr) == 0:
        return arr

    # Identify zero scores (failed I0 threshold)
    zero_mask = arr == 0.0
    non_zero = arr[~zero_mask]

    if len(non_zero) == 0:
        return arr  # all zero

    min_val = np.min(non_zero)
    max_val = np.max(non_zero)

    if max_val == min_val:
        # All non-zero equal → assign 0.5
        arr[~zero_mask] = 0.5
    else:
        arr[~zero_mask] = (non_zero - min_val) / (max_val - min_val)

    return arr


# ============================================================================
#                           MAIN EVALUATION
# ============================================================================

def evaluate_population_fitness(directory, deut_ref=None, prot_ref=None,
                                q_max=0.3, i0_threshold=0.2):
    """
    Evaluate fitness for all .dat files in the given directory.

    Args:
        directory (str): Path to directory with simulated .dat files and ref/
        deut_ref (str, optional): Exact deuterated reference filename
        prot_ref (str, optional): Exact protonated reference filename
        q_max (float): q truncation limit
        i0_threshold (float): Minimum I(0) ratio

    Returns:
        tuple: (normalized_fitness, file_paths)
    """
    # --- Locate data files ---
    sim_files = find_dat_files(directory)
    logger.info(f"Found {len(sim_files)} simulated .dat files")

    ref_dir = os.path.join(directory, "ref")
    deut_path, prot_path = find_reference_files(ref_dir, deut_ref, prot_ref)

    # --- Load reference curves ---
    q_deut, I_deut = parse_sans_file(deut_path)
    q_prot, I_prot = parse_sans_file(prot_path)
    logger.info(f"Loaded deuterated reference: {os.path.basename(deut_path)} [{len(q_deut)} points]")
    logger.info(f"Loaded protonated reference: {os.path.basename(prot_path)} [{len(q_prot)} points]")

    # --- Process each simulation file ---
    raw_scores = []
    valid_files = []

    for file_path in sim_files:
        try:
            q, I = parse_sans_file(file_path)
            score = fitness(q, I, I_deut, I_prot, q_max, i0_threshold)
            raw_scores.append(score)
            valid_files.append(file_path)
            logger.debug(f"{os.path.basename(file_path)}: raw score = {score:.6e}")
        except Exception as e:
            logger.warning(f"Skipping {os.path.basename(file_path)}: {e}")
            raw_scores.append(0.0)
            valid_files.append(file_path)

    # --- Normalize ---
    normalized = normalize_fitness(raw_scores)

    return normalized, valid_files


def main():
    args = parse_arguments()

    try:
        fitness_scores, sim_files = evaluate_population_fitness(
            directory=args.directory,
            deut_ref=args.deut_ref,
            prot_ref=args.prot_ref,
            q_max=args.q_max,
            i0_threshold=args.i0_threshold
        )
    except (NotADirectoryError, FileNotFoundError, ValueError) as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

    # --- Output ---
    if not args.quiet:
        print("\n" + "=" * 60)
        print("FITNESS EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Directory: {args.directory}")
        print(f"Q max: {args.q_max:.3f} Å⁻¹")
        print(f"I0 threshold: {args.i0_threshold:.2f} × I0(prot)")
        print(f"Files evaluated: {len(fitness_scores)}")
        print(f"Files passing I0 threshold: {np.sum(fitness_scores > 0)}/{len(fitness_scores)}")
        print(f"Best fitness: {np.max(fitness_scores):.6f}")
        if len(fitness_scores) > 0 and np.max(fitness_scores) > 0:
            best_idx = np.argmax(fitness_scores)
            print(f"Best file: {os.path.basename(sim_files[best_idx])}")
        print(f"Average fitness: {np.mean(fitness_scores):.6f}")
        print("=" * 60 + "\n")

    # Always print the normalized fitness array (machine-readable)
    # Format: one float per line
    for score in fitness_scores:
        print(f"{score:.6f}")


if __name__ == "__main__":
    import sys
    main()
