#!/usr/bin/env python3
"""
PDB Deuteration Module
======================

Standalone module for deuterating protein structures in PDB format.

This module can:
- Convert hydrogen atoms (H) to deuterium (D) based on amino acid selection
- Apply D2O exchange to labile hydrogens
- Be used standalone via command line or imported as a library

Usage (Standalone):
    python pdb_deuteration.py input.pdb config.ini output.pdb
    python pdb_deuteration.py config.ini
    python pdb_deuteration.py -i input.pdb -o output.pdb --d2o 80 --ALA --GLY
    
Usage (Library):
    from pdb_deuteration import PdbDeuteration, AMINO_ACIDS
    deuterator = PdbDeuteration("input.pdb")
    deuterator.apply_deuteration(deuteration_vector, d2o_percent)
    deuterator.save("output.pdb")
"""

import sys
import gemmi
import random
import logging
import argparse
import configparser
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass


# ============================================================================
#                           LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
#                           AMINO ACID DEFINITIONS
# ============================================================================

@dataclass
class AminoAcid:
    """
    Represents an amino acid with its nomenclature codes.

    Attributes:
        name (str): Full name of the amino acid (e.g., "Alanine")
        code_3 (str): 3-letter code (e.g., "ALA")
        code_1 (str): 1-letter code (e.g., "A")
    """
    name: str
    code_3: str
    code_1: str


# List of standard amino acids
AMINO_ACIDS = [
    AminoAcid("Alanine", "ALA", "A"),
    AminoAcid("Arginine", "ARG", "R"),
    AminoAcid("Asparagine", "ASN", "N"),
    AminoAcid("Aspartic acid", "ASP", "D"),
    AminoAcid("Cysteine", "CYS", "C"),
    AminoAcid("Glutamic acid", "GLU", "E"),
    AminoAcid("Glutamine", "GLN", "Q"),
    AminoAcid("Glycine", "GLY", "G"),
    AminoAcid("Histidine", "HIS", "H"),
    AminoAcid("Isoleucine", "ILE", "I"),
    AminoAcid("Leucine", "LEU", "L"),
    AminoAcid("Lysine", "LYS", "K"),
    AminoAcid("Methionine", "MET", "M"),
    AminoAcid("Phenylalanine", "PHE", "F"),
    AminoAcid("Proline", "PRO", "P"),
    AminoAcid("Serine", "SER", "S"),
    AminoAcid("Threonine", "THR", "T"),
    AminoAcid("Tryptophan", "TRP", "W"),
    AminoAcid("Tyrosine", "TYR", "Y"),
    AminoAcid("Valine", "VAL", "V")
]

# Dictionary for quick access by 3-letter code
AA_DICT = {aa.code_3: aa for aa in AMINO_ACIDS}
AA_INDEX = {aa.code_3: i for i, aa in enumerate(AMINO_ACIDS)}


# ============================================================================
#                           ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """
    Parses command-line arguments for PDB deuteration.

    Returns:
        argparse.Namespace: Parsed arguments

    """
    parser = argparse.ArgumentParser(
        description="Deuterate protein structures in PDB format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Using config file only
  python pdb_deuteration.py pdb_config.ini
  
  # Full command line mode
  python pdb_deuteration.py -i input.pdb -o output.pdb --d2o 50 --ALA --GLY --LEU
  
  # Mixed mode (CLI overrides INI)
  python pdb_deuteration.py pdb_config.ini --d2o 80 --ILE --MET
  
  # Deuterate all amino acids
  python pdb_deuteration.py -i input.pdb -o output.pdb --all --d2o 100
  
  # Show help
  python pdb_deuteration.py -h
        """
    )

    # Optional config file (positional)
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to configuration INI file (optional if all parameters provided via CLI)"
    )

    # I/O PARAMETERS
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        '-i', '--input_pdb',
        type=str,
        help='Input PDB file path. Default: from config file or required'
    )
    io_group.add_argument(
        '-o', '--output_pdb',
        type=str,
        help='Output PDB file path. Default: from config file or "deuterated.pdb"'
    )

    # DEUTERATION PARAMETERS 
    deut_group = parser.add_argument_group("Deuteration parameters")
    deut_group.add_argument(
        '--d2o', '--d2o_percent',
        dest='d2o_percent',
        type=float,
        help='D2O percentage for labile hydrogen exchange (0-100). Default: 0'
    )

    # AMINO ACID SELECTION
    aa_group = parser.add_argument_group("Amino acid selection (mutually exclusive with --all)")
    
    # Create mutually exclusive group for AA selection vs --all
    aa_selector = aa_group.add_mutually_exclusive_group()
    
    aa_selector.add_argument(
        '--all',
        action='store_true',
        help='Deuterate ALL amino acid types (overrides individual selections)'
    )
    
    # Individual amino acid flags (store_true = deuterate this AA)
    for aa in AMINO_ACIDS:
        aa_group.add_argument(
            f'--{aa.code_3}',
            action='store_true',
            help=f'Deuterate {aa.name} ({aa.code_3})'
        )
    
    # Negative selection (store_false = do NOT deuterate this AA)
    for aa in AMINO_ACIDS:
        aa_group.add_argument(
            f'--no-{aa.code_3}',
            action='store_false',
            dest=f'{aa.code_3}',
            help=f'Do NOT deuterate {aa.name} ({aa.code_3})'
        )

    # VERBOSITY
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (debug mode)'
    )

    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    return args


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Validates the configuration parameters.

    Args:
        cfg: Configuration dictionary

    Raises:
        ValueError: If any parameter is invalid
    """
    # I/O validation
    if not cfg.get("input_pdb"):
        raise ValueError("input_pdb is required")
    
    if not cfg.get("output_pdb"):
        cfg["output_pdb"] = "deuterated.pdb"
        logger.info(f"output_pdb not set, using default: {cfg['output_pdb']}")

    # D2O percentage validation
    if not 0 <= cfg["d2o_percent"] <= 100:
        raise ValueError(
            f"d2o_percent must be in [0, 100], got {cfg['d2o_percent']}"
        )

    # Restrictions validation
    if len(cfg["deuteration_vector"]) != len(AMINO_ACIDS):
        raise ValueError(
            f"deuteration_vector length ({len(cfg['deuteration_vector'])}) "
            f"!= number of amino acids ({len(AMINO_ACIDS)})"
        )


def load_config_ini(path: str) -> Dict[str, Any]:
    """
    Loads deuteration configuration from an INI file.

    Args:
        path: Path to configuration file

    Returns:
        Dictionary with configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is malformed
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = configparser.ConfigParser()
    config.read(path)

    cfg = {}

    # Read deuteration parameters (with fallbacks)
    try:
        cfg["input_pdb"] = config.get("DEUTERATION", "input_pdb", fallback=None)
        cfg["output_pdb"] = config.get("DEUTERATION", "output_pdb", fallback=None)
        cfg["d2o_percent"] = config.getfloat("DEUTERATION", "d2o_percent", fallback=0.0)
    except (configparser.NoSectionError, ValueError) as e:
        raise ValueError(f"Error reading [DEUTERATION] section: {e}")

    # Read amino acid selection (default to False if not specified)
    try:
        deuteration_vector = []
        for aa in AMINO_ACIDS:
            # If section exists, try to get value, default to False
            if config.has_section("AMINO_ACIDS"):
                value = config.getboolean("AMINO_ACIDS", aa.code_3, fallback=False)
            else:
                value = False
            deuteration_vector.append(value)
        cfg["deuteration_vector"] = deuteration_vector
    except ValueError as e:
        raise ValueError(f"Error reading [AMINO_ACIDS] section: {e}")

    return cfg


def merge_config(cli_args: argparse.Namespace, ini_cfg: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Merges CLI arguments with INI configuration (CLI takes precedence).

    Args:
        cli_args: Parsed command-line arguments
        ini_cfg: Configuration loaded from INI file (optional)

    Returns:
        Merged configuration dictionary
    """
    ini_cfg = ini_cfg or {}
    
    def pick(cli_val, ini_val, default):
        """Pick CLI value if not None, otherwise INI value, otherwise default."""
        return cli_val if cli_val is not None else (ini_val if ini_val is not None else default)

    # Build deuteration vector from CLI flags
    cli_deuteration_vector = None
    if cli_args.all:
        # --all overrides everything: deuterate all AAs
        cli_deuteration_vector = [True] * len(AMINO_ACIDS)
    else:
        # Check if any AA was explicitly set via CLI
        has_aa_selection = any(getattr(cli_args, aa.code_3) is not None for aa in AMINO_ACIDS)
        if has_aa_selection:
            cli_deuteration_vector = []
            for aa in AMINO_ACIDS:
                cli_value = getattr(cli_args, aa.code_3)
                # If explicitly set, use CLI value, otherwise keep INI/default
                if cli_value is not None:
                    cli_deuteration_vector.append(cli_value)
                else:
                    # Not set in CLI, will use INI value later
                    cli_deuteration_vector = None
                    break

    # Merge deuteration vector
    if cli_deuteration_vector is not None:
        # CLI completely defines the vector
        deuteration_vector = cli_deuteration_vector
    else:
        # Use INI vector or default
        ini_vector = ini_cfg.get("deuteration_vector", [False] * len(AMINO_ACIDS))
        deuteration_vector = []
        for i, aa in enumerate(AMINO_ACIDS):
            cli_value = getattr(cli_args, aa.code_3)
            if cli_value is not None:
                # CLI overrides individual AA
                deuteration_vector.append(cli_value)
            else:
                # Use INI value
                deuteration_vector.append(ini_vector[i])

    return {
        # I/O
        "input_pdb": pick(cli_args.input_pdb, ini_cfg.get("input_pdb"), None),
        "output_pdb": pick(cli_args.output_pdb, ini_cfg.get("output_pdb"), "deuterated.pdb"),
        
        # Deuteration
        "d2o_percent": pick(cli_args.d2o_percent, ini_cfg.get("d2o_percent"), 0.0),
        
        # Amino acid selection
        "deuteration_vector": deuteration_vector,
        
        # Metadata
        "config_file": getattr(cli_args, "config", None),
        "verbose": cli_args.verbose
    }


# ============================================================================
#                           LABILITY HELPER
# ============================================================================

def _compute_lability_for_residue(residue: gemmi.Residue) -> List[bool]:
    """
    Determine for each atom in a residue whether it is a labile hydrogen.

    A hydrogen (or deuterium) is labile when it is bonded to O, N, or S.
    This is approximated by the ordering in the PDB file: an H/D atom is
    labile if the immediately preceding non-H/D atom is O, N, or S.

    This is the same logic as the original ``_is_labile_hydrogen``; it is
    factored out as a plain function so it can be called during cache
    construction without instantiating a PdbDeuteration object.

    Args:
        residue: A gemmi Residue object.

    Returns:
        List[bool]: One boolean per atom in the residue.
                    True  → atom is a labile H or D.
                    False → atom is not labile (heavy atom, non-labile H/D, …).
    """
    labile_list: List[bool] = []
    preceding_is_heteroatom = False  # True when last heavy atom was O, N or S

    for atom in residue:
        elem = atom.element.name
        if elem in ("O", "N", "S"):
            labile_list.append(False)
            preceding_is_heteroatom = True
        elif elem in ("H", "D"):
            labile_list.append(preceding_is_heteroatom)
            # Do NOT reset preceding_is_heteroatom: consecutive H on the same
            # heteroatom (e.g. NH2) are all labile.
        else:
            # Carbon or other heavy atom: next H will not be labile.
            preceding_is_heteroatom = False
            labile_list.append(False)

    return labile_list


# ============================================================================
#                           PDB DEUTERATION CLASS
# ============================================================================

class PdbDeuteration:
    """
    Main class for deuteration of PDB files.
    
    This class handles:
    - Loading PDB structures
    - Converting H to D atoms based on amino acid selection
    - Applying D2O exchange to labile hydrogens
    - Saving modified structures

    Optimization notes
    ------------------
    The lability map (which atoms are labile H/D) is computed **once** during
    ``__init__`` and stored in ``self._lability_cache``.  Subsequent calls to
    ``apply_deuteration`` read from the cache instead of re-computing lability
    for every residue.

    ``apply_deuteration`` performs a **single pass** over the atom list:
    it simultaneously applies the H→D conversion and updates the atom counters
    in ``self.stats``.  The original design required two passes
    (one for deuteration, one for ``_count_atoms``).

    Attributes:
        pdb_path (Path)           : Path to the PDB file
        structure (gemmi.Structure): Parsed PDB structure
        stats (dict)              : Counters for H/D atoms (updated after each call)
        _lability_cache (dict)    : Pre-computed per-residue lability vectors
                                    keyed by (model_idx, chain_idx, residue_idx)
    """

    def __init__(self, pdb_file: str):
        """
        Initialize the deuterator with a PDB file.

        The lability map is computed here once (single pass over all atoms),
        and the initial atom counts are derived from that same pass.

        Args:
            pdb_file: Path to the PDB file.

        Raises:
            FileNotFoundError: If PDB file doesn't exist
            RuntimeError: If PDB parsing fails
        """
        self.pdb_path = Path(pdb_file)

        if not self.pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        try:
            self.structure = gemmi.read_structure(str(self.pdb_path))
            logger.debug(f"Structure loaded: {self.pdb_path.name}")
            logger.debug(f"Models: {len(self.structure)}, "
                        f"Chains: {sum(len(model) for model in self.structure)}")
        except Exception as e:
            raise RuntimeError(f"Error while parsing PDB: {e}")

        # Statistics (kept identical to original for full compatibility)
        self.stats: Dict[str, int] = {
            'total_atoms':    0,
            'hydrogen_atoms': 0,
            'deuterium_atoms':0,
            'labile_H':       0,
            'labile_D':       0,
            'non_labile_H':   0,
            'non_labile_D':   0,
        }

        # Lability cache: (model_idx, chain_idx, residue_idx) -> List[bool]
        # Computed once; remains valid across multiple apply_deuteration calls
        # because lability depends only on surrounding heavy-atom types (O/N/S/C),
        # which are never modified by deuteration.
        self._lability_cache: Dict[Tuple[int, int, int], List[bool]] = {}

        # Build the lability cache AND compute initial atom counts in one pass.
        self._build_lability_cache_and_count()

    # ------------------------------------------------------------------
    #  INTERNAL: one-time setup
    # ------------------------------------------------------------------

    def _build_lability_cache_and_count(self) -> None:
        """
        Build ``self._lability_cache`` and initialise ``self.stats`` in a
        single traversal of the structure.

        This replaces the original separate calls to
        ``_count_atoms()`` (which itself called ``_is_labile_hydrogen`` per
        residue) and avoids iterating over the atom list twice.
        """
        # Reset counters
        for key in self.stats:
            self.stats[key] = 0

        for mi, model in enumerate(self.structure):
            for ci, chain in enumerate(model):
                for ri, residue in enumerate(chain):
                    # Compute lability once for this residue
                    labile_vector = _compute_lability_for_residue(residue)
                    self._lability_cache[(mi, ci, ri)] = labile_vector

                    # Count atoms using the freshly computed lability vector
                    for atom, is_labile in zip(residue, labile_vector):
                        elem = atom.element.name
                        self.stats['total_atoms'] += 1

                        if elem == "H":
                            self.stats['hydrogen_atoms'] += 1
                            if is_labile:
                                self.stats['labile_H'] += 1
                            else:
                                self.stats['non_labile_H'] += 1

                        elif elem == "D":
                            self.stats['deuterium_atoms'] += 1
                            if is_labile:
                                self.stats['labile_D'] += 1
                            else:
                                self.stats['non_labile_D'] += 1

    # ------------------------------------------------------------------
    #  PUBLIC: deuteration
    # ------------------------------------------------------------------

    def apply_deuteration(self,
                          deuteration_vector: List[bool],
                          d2o_percent: int = 0) -> None:
        """
        Apply deuteration to the PDB structure.

        Performs a **single pass** over all atoms:
          1. Converts non-labile H → D for selected amino acid types.
          2. Probabilistically converts labile H → D based on *d2o_percent*.
          3. Updates ``self.stats`` in the same pass (no second iteration).

        The lability information is read from the pre-computed
        ``self._lability_cache`` (populated during ``__init__``).

        Args:
            deuteration_vector: List of 20 booleans (one per amino acid in
                                AMINO_ACIDS order).  True = deuterate all
                                non-labile H in that AA type.
            d2o_percent:        Percentage of D2O for labile hydrogen exchange
                                (integer 0-100).

        Raises:
            ValueError: If *deuteration_vector* doesn't have exactly 20 elements.
            ValueError: If *d2o_percent* is not in [0, 100].
        """
        if len(deuteration_vector) != 20:
            raise ValueError(
                f"deuteration_vector must contain 20 elements, "
                f"{len(deuteration_vector)} received"
            )
        if not 0 <= d2o_percent <= 100:
            raise ValueError(
                f"d2o_percent must be between 0 and 100, "
                f"{d2o_percent} received"
            )

        logger.debug(f"Applying deuteration: D₂O = {d2o_percent}%")
        deuterated_aas = [
            aa.code_3 for aa, deut in zip(AMINO_ACIDS, deuteration_vector) if deut
        ]
        if deuterated_aas:
            logger.debug(f"Deuterated amino acids: {', '.join(deuterated_aas)}")
        else:
            logger.debug(
                "No amino acids selected for deuteration "
                "(non-labile H will remain H)"
            )

        # Reset counters — they are recomputed during this single pass
        for key in self.stats:
            self.stats[key] = 0

        for mi, model in enumerate(self.structure):
            for ci, chain in enumerate(model):
                for ri, residue in enumerate(chain):
                    residue_name = residue.name.strip().upper()

                    # ---- Determine deuteration intent for this residue ----
                    aa_index       = AA_INDEX.get(residue_name)     # None for non-AA
                    should_deut_aa = (
                        aa_index is not None and deuteration_vector[aa_index]
                    )

                    # Read pre-computed lability vector (always available)
                    labile_vector = self._lability_cache[(mi, ci, ri)]

                    # ---- Single pass: modify + count ----
                    for atom, is_labile in zip(residue, labile_vector):
                        elem = atom.element.name

                        # --- Possible H→D conversions ---
                        if elem == "H":
                            if is_labile:
                                # D2O exchange: probabilistic
                                if random.random() * 100 < d2o_percent:
                                    self._convert_atom_H_to_D(atom)
                                    elem = "D"  # update local variable for counting
                            elif should_deut_aa and aa_index is not None:
                                # Non-labile H in a selected AA type
                                self._convert_atom_H_to_D(atom)
                                elem = "D"  # update local variable for counting

                        # --- Count (using potentially updated element) ---
                        self.stats['total_atoms'] += 1

                        if elem == "H":
                            self.stats['hydrogen_atoms'] += 1
                            if is_labile:
                                self.stats['labile_H'] += 1
                            else:
                                self.stats['non_labile_H'] += 1

                        elif elem == "D":
                            self.stats['deuterium_atoms'] += 1
                            if is_labile:
                                self.stats['labile_D'] += 1
                            else:
                                self.stats['non_labile_D'] += 1

    # ------------------------------------------------------------------
    #  INTERNAL: atom conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_atom_H_to_D(atom: gemmi.Atom) -> None:
        """
        Convert a hydrogen atom to deuterium (in-place).

        Args:
            atom: The atom to modify (must have element H).
        """
        atom.name    = atom.name.replace('H', 'D', 1)
        atom.element = gemmi.Element("D")

    # ------------------------------------------------------------------
    #  KEPT FOR COMPATIBILITY — wraps the cache-based approach
    # ------------------------------------------------------------------

    def _is_labile_hydrogen(self, residue: gemmi.Residue) -> List[bool]:
        """
        Determine for each atom in a residue if it's a labile hydrogen.

        .. note::
            This method is retained for API compatibility.  Internally,
            ``PdbDeuteration`` now uses the pre-computed ``_lability_cache``
            instead of calling this method repeatedly.  External code that
            calls this method will still get correct results but will not
            benefit from the caching.

        Args:
            residue: Residue to analyze.

        Returns:
            List[bool]: One boolean per atom in the residue.
        """
        return _compute_lability_for_residue(residue)

    def _count_atoms(self) -> None:
        """
        Recount all atoms and update ``self.stats``.

        .. note::
            This method is retained for API compatibility.  In the optimized
            version, counting is done inside ``apply_deuteration`` so an
            explicit call to ``_count_atoms`` is never needed.  Calling it
            externally after ``apply_deuteration`` is safe but redundant.
        """
        for key in self.stats:
            self.stats[key] = 0

        for mi, model in enumerate(self.structure):
            for ci, chain in enumerate(model):
                for ri, residue in enumerate(chain):
                    labile_vector = self._lability_cache.get(
                        (mi, ci, ri),
                        _compute_lability_for_residue(residue)  # fallback
                    )
                    for atom, is_labile in zip(residue, labile_vector):
                        elem = atom.element.name
                        self.stats['total_atoms'] += 1
                        if elem == "H":
                            self.stats['hydrogen_atoms'] += 1
                            if is_labile:
                                self.stats['labile_H'] += 1
                            else:
                                self.stats['non_labile_H'] += 1
                        elif elem == "D":
                            self.stats['deuterium_atoms'] += 1
                            if is_labile:
                                self.stats['labile_D'] += 1
                            else:
                                self.stats['non_labile_D'] += 1

    # ------------------------------------------------------------------
    #  PUBLIC: save
    # ------------------------------------------------------------------

    def save(self, output_path: str) -> None:
        """
        Save the modified structure to a PDB file.
        
        Args:
            output_path: Path to the output file
            
        Raises:
            IOError: If writing fails
        """
        try:
            self.structure.write_pdb(str(Path(output_path)))
            logger.debug(f"Structure saved: {output_path}")
        except Exception as e:
            raise IOError(f"Error while saving: {e}")




# ============================================================================
#                           MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for standalone usage."""
    # Parse command line arguments
    cli_args = parse_arguments()

    # Load INI config if provided
    ini_cfg = None
    if cli_args.config:
        try:
            ini_cfg = load_config_ini(cli_args.config)
            logger.info(f"Loaded configuration from: {cli_args.config}")
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error loading config file: {e}")
            sys.exit(1)

    # Merge CLI and INI configurations
    try:
        config = merge_config(cli_args, ini_cfg)
        validate_config(config)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Display configuration summary
    logger.info("=" * 60)
    logger.info("PDB DEUTERATION")
    logger.info("=" * 60)
    logger.info(f"Input PDB:  {config['input_pdb']}")
    logger.info(f"Output PDB: {config['output_pdb']}")
    logger.info(f"D2O:        {config['d2o_percent']:.2f}%")
    
    deuterated_count = sum(config['deuteration_vector'])
    logger.info(f"Amino acids: {deuterated_count}/20 selected for deuteration")
    
    if config['config_file']:
        logger.info(f"Config file: {config['config_file']}")
    logger.info("=" * 60)

    # Load and deuterate
    try:
        deuterator = PdbDeuteration(config['input_pdb'])
        deuterator.apply_deuteration(
            config['deuteration_vector'],
            config['d2o_percent']
        )
        print(f"Total Atoms              : {deuterator.stats['total_atoms']}")
        print(f"Hydrogen atoms           : {deuterator.stats['hydrogen_atoms']}")
        print(f"Labile Hydrogen atoms    : {deuterator.stats['labile_H']}")
        print(f"Non Labile Hydrogen atoms: {deuterator.stats['non_labile_H']}")
        print(f"Deuterium atoms          : {deuterator.stats['deuterium_atoms']}")
        print(f"Labile Deuterium atoms   : {deuterator.stats['labile_D']}")
        print(f"Non Labile Deuterium atoms: {deuterator.stats['non_labile_D']}")

        deuterator.save(config['output_pdb'])
    except (FileNotFoundError, RuntimeError, IOError, ValueError) as e:
        logger.error(f"Deuteration failed: {e}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Deuteration completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
