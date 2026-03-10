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
import re
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

Curve validity check:
  A curve is accepted only if ratio = Imax / background > 0.01
  where background = -0.0117 * d2o_percent + 1.25
  and d2o_percent is extracted from the filename
  (e.g. genXX_chrXXX_d2o42_deutAAXX.dat -> D2O = 42%).

Examples:
  # Basic usage (assumes default naming in ref/)
  python fitness_evaluation.py /path/to/simulations

  # Specify custom reference filenames
  python fitness_evaluation.py /path/to/simulations --deut-ref D2O.dat --prot-ref H2O.dat

  # Adjust q-range and ratio threshold
  python fitness_evaluation.py /path/to/simulations --q-max 0.35 --ratio-threshold 0.05

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
        help='Filename of deuterated reference inside ref/ (default: auto-detect with "*deuteration.dat")'
    )
    ref_group.add_argument(
        '--prot-ref',
        type=str,
        default=None,
        help='Filename of protonated reference inside ref/ (default: auto-detect with "*protonation.dat")'
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
        '--ratio-threshold',
        type=float,
        default=0.01,
        help='Minimum Imax/background ratio to accept a curve (default: 0.01)'
    )
    # Kept for backward compatibility with older call sites, but no longer used
    param_group.add_argument(
        '--i0-threshold',
        type=float,
        default=None,
        help=argparse.SUPPRESS
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
        help='Suppress all non-essential output (only print final fitness array)'
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


# ============================================================================
#                     RATIO CHECK (replaces i0_threshold_check)
# ============================================================================

def extract_d2o_from_filename(filename):
    """
    Extract the D2O percentage from a curve filename.

    Expected filename pattern: genXX_chrXXX_d2oYY_deutAAZZ.dat (or .out)
    Example: gen01_chr003_d2o42_deutAA05.dat -> returns 42

    Args:
        filename (str): Basename or full path of the curve file

    Returns:
        int or None: D2O percentage, or None if not found in filename
    """
    basename = os.path.basename(filename)
    match = re.search(r'_d2o(\d+)', basename)
    if match:
        return int(match.group(1))
    return None


def compute_incoherent_background(d2o_percent):
    """
    Compute the incoherent scattering background for a given D2O percentage.

    Formula: background = -0.0117 * d2o_percent + 1.25

    Args:
        d2o_percent (int or float): D2O percentage (0-100)

    Returns:
        float: Background value
    """
    return -0.0117 * d2o_percent + 1.25


def ratio_check(I, d2o_percent, threshold=0.01):
    """
    Check if a curve is valid based on the Imax / background ratio.

    A curve is accepted if:
        ratio = Imax / background > threshold
    where:
        Imax       = maximum value of I (not necessarily I[0])
        background = -0.0117 * d2o_percent + 1.25

    Args:
        I (np.array): Simulated intensity array
        d2o_percent (int or float): D2O percentage extracted from the filename
        threshold (float): Minimum ratio to accept the curve (default: 0.01)

    Returns:
        bool: True if the curve passes the ratio check, False otherwise
    """
    background = compute_incoherent_background(d2o_percent)
    if background <= 0:
        logger.warning(
            f"Non-positive background ({background:.4f}) for D2O={d2o_percent}%; "
            f"curve rejected"
        )
        return 0.0

    I_max = np.max(I)
    ratio = I_max / background

    logger.debug(
        f"Ratio check: Imax={I_max:.4e}, background={background:.4f}, "
        f"ratio={ratio:.4f}, threshold={threshold}"
    )

    return ratio


def scaling_and_compare(q, I, I_deut, I_prot, q_max, ratio):
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
    # to compare to the 2 curves return area_deut + area_prot
    return area_deut * area_prot * ratio * 10000


def fitness(q, I, I_deut, I_prot, q_max, file_path, ratio_threshold=0.01):
    """
    Compute raw fitness score (area difference) for a single curve.

    The curve is validated by checking that ratio = Imax / background > ratio_threshold,
    where background = -0.0117 * d2o_percent + 1.25 and d2o_percent is extracted
    from the filename (pattern: genXX_chrXXX_d2oYY_deutAAZZ.dat or .out).
    If d2o_percent cannot be determined from the filename, the ratio check is skipped
    and the curve is accepted.

    Args:
        q (np.array): q values
        I (np.array): Simulated intensity
        I_deut (np.array): Deuterated reference
        I_prot (np.array): Protonated reference
        q_max (float): q truncation limit
        file_path (str): Path to the curve file (used to extract D2O percentage)
        ratio_threshold (float): Minimum Imax/background ratio (default: 0.01)

    Returns:
        float: Raw area sum, or 0.0 if ratio check fails
    """
    d2o_percent = extract_d2o_from_filename(file_path)

    if d2o_percent is None:
        logger.debug(
            f"Could not extract D2O from filename '{os.path.basename(file_path)}'; "
            f"ratio check skipped, curve accepted by default"
        )
        ratio = 0.0
    else:
        ratio = ratio_check(I, d2o_percent, threshold=ratio_threshold)
        if ratio <ratio_threshold:
            logger.debug(
                f"Ratio check failed for '{os.path.basename(file_path)}': "
                f"ratio={ratio:.4f} < threshold={ratio_threshold}"
            )
            #print(
            #    f"Ratio check FAILED for '{os.path.basename(file_path)}': "
            #    f"ratio={ratio:.4f} < threshold={ratio_threshold}")
            return 0.0, ratio
        else :
            logger.debug(
                f"Ratio check passed for '{os.path.basename(file_path)}': "
                f"ratio={ratio:.4f} > threshold={ratio_threshold}"
            )
            #print(
            #    f"Ratio check passed for '{os.path.basename(file_path)}': "
            #f"ratio={ratio:.4f} > threshold={ratio_threshold}")
    return scaling_and_compare(q, I, I_deut, I_prot, q_max, ratio), ratio


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

    # Identify zero scores (failed ratio check)
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
                                q_max=0.3, i0_threshold=None, ratio_threshold=0.01):
    """
    Evaluate fitness for all .dat files in the given directory.

    Args:
        directory (str): Path to directory with simulated .dat files and ref/
        deut_ref (str, optional): Exact deuterated reference filename
        prot_ref (str, optional): Exact protonated reference filename
        q_max (float): q truncation limit
        i0_threshold (float, optional): Deprecated, kept for backward compatibility (ignored)
        ratio_threshold (float): Minimum Imax/background ratio to accept a curve

    Returns:
        tuple: (raw_fitness_scores, file_paths)
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
    ratios = []

    for file_path in sim_files:
        try:
            q, I = parse_sans_file(file_path)
            score, ratio = fitness(q, I, I_deut, I_prot, q_max, file_path, ratio_threshold)
            #score= fitness(q, I, I_deut, I_prot, q_max, file_path, ratio_threshold)
            raw_scores.append(score)
            valid_files.append(file_path)
            ratios.append(ratio)
            #print(f"THIS IS RATIO IN FITNESS CODE {ratio} and fitness of {score}")
            logger.debug(f"{os.path.basename(file_path)}: ratio = {ratio:.6e} raw score = {score:.6e}")
        except Exception as e:
            logger.warning(f"Skipping {os.path.basename(file_path)}: {e}")
            raw_scores.append(0.0)
            valid_files.append(file_path)

    # --- Normalize ---
    normalized = normalize_fitness(raw_scores)

    #return normalized, valid_files
    #test returning only raw scores
    return np.array(raw_scores, dtype=float), valid_files, np.array(ratios, dtype=float)


def main():
    args = parse_arguments()

    try:
        #fitness_scores, sim_files = evaluate_population_fitness(
        fitness_scores, sim_files, ratios = evaluate_population_fitness(
            directory=args.directory,
            deut_ref=args.deut_ref,
            prot_ref=args.prot_ref,
            q_max=args.q_max,
            ratio_threshold=args.ratio_threshold
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
        print(f"Ratio threshold (Imax/background): {args.ratio_threshold}")
        print(f"Files evaluated: {len(fitness_scores)}")
        print(f"Files passing ratio check: {np.sum(fitness_scores > 0)}/{len(fitness_scores)}")
        print(f"Best fitness: {np.max(fitness_scores):.8f}")
        if len(fitness_scores) > 0 and np.max(fitness_scores) > 0:
            best_idx = np.argmax(fitness_scores)
            print(f"Best file: {os.path.basename(sim_files[best_idx])}")
        print(f"Average fitness: {np.mean(fitness_scores):.8f}")
        print("=" * 60 + "\n")

    # Always print the normalized fitness array (machine-readable)
    # Format: one float per line
    for score in fitness_scores:
        print(f"{score:.8f}")


if __name__ == "__main__":
    import sys
    main()
