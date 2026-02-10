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
    python pdb_deuteration.py input.pdb pdb_config.ini output.pdb
    
Usage (Library):
    from pdb_deuteration import PdbDeuteration, AMINO_ACIDS
    deuterator = PdbDeuteration("input.pdb")
    deuterator.apply_deuteration(deuteration_vector, d2o_percent)
    deuterator.save("output.pdb")

Author: Your Name
Date: 2025
"""

import sys
import gemmi
import random
import logging
import configparser
from pathlib import Path
from typing import List
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
AA_DICT = {aa.code_3: i for i, aa in enumerate(AMINO_ACIDS)}


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
    
    Attributes:
        pdb_path (Path): Path to the PDB file
        structure (gemmi.Structure): Parsed PDB structure
        stats (dict): Statistics about atoms in the structure
    """
    
    def __init__(self, pdb_file: str):
        """
        Initialize the deuterator with a PDB file.
        
        Args:
            pdb_file: Path to the PDB file
            
        Raises:
            FileNotFoundError: If PDB file doesn't exist
            RuntimeError: If PDB parsing fails
        """
        self.pdb_path = Path(pdb_file)

        if not self.pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        try:
            self.structure = gemmi.read_structure(str(self.pdb_path))
            logger.info(f"Structure loaded: {self.pdb_path.name}")
            logger.info(f"Models: {len(self.structure)}, "
                       f"Chains: {sum(len(model) for model in self.structure)}")
        except Exception as e:
            raise RuntimeError(f"Error while parsing PDB: {e}")

        # Statistics for validation
        self.stats = {
            'total_atoms': 0,
            'hydrogen_atoms': 0,
            'deuterium_atoms': 0,
            'labile_H': 0,
            'non_labile_H': 0
        }
        self._count_atoms()

    def _count_atoms(self) -> None:
        """Count all atoms, H atoms, and D atoms for statistics."""
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        self.stats['total_atoms'] += 1
                        if atom.element.name == "H":
                            self.stats['hydrogen_atoms'] += 1
                        elif atom.element.name == "D":
                            self.stats['deuterium_atoms'] += 1

    def apply_deuteration(self,
                          deuteration_vector: List[bool],
                          d2o_percent: float = 0.0) -> None:
        """
        Apply deuteration to the PDB structure.
        
        Args:
            deuteration_vector: List of 20 booleans (one per amino acid)
                               True = deuterate all non-labile H in this AA type
            d2o_percent: Percentage of D2O for labile hydrogen exchange (0-100)
            
        Raises:
            ValueError: If deuteration_vector doesn't have exactly 20 elements
            ValueError: If d2o_percent is not in range [0, 100]
        """
        # Input validation
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

        logger.info(f"Applying deuteration: D₂O = {d2o_percent:.2f}%")
        logger.info(f"Deuterated amino acids: {sum(deuteration_vector)}/20")

        # Iterate through the structure
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    residue_name = residue.name.strip().upper()

                    # Check if it's a standard amino acid
                    if residue_name not in AA_DICT:
                        logger.debug(f"Non-standard residue ignored: {residue_name}")
                        continue

                    # Determine lability for each atom in the residue
                    labile_vector = self._is_labile_hydrogen(residue)
                    
                    # Get amino acid index
                    aa_index = AA_DICT[residue_name]
                    should_deuterate = deuteration_vector[aa_index]

                    # Process each atom in the residue
                    for atom, is_labile in zip(residue, labile_vector):
                        element = atom.element.name

                        # Deuterate non-labile H if this AA type is selected
                        if should_deuterate:
                            if element == "H" and not is_labile:
                                self._convert_atom_H_to_D(atom)

                        # Apply D2O exchange to labile H
                        if is_labile:
                            if random.random() * 100 < d2o_percent:
                                if element == "H":
                                    self._convert_atom_H_to_D(atom)

    def _convert_atom_H_to_D(self, atom: gemmi.Atom) -> None:
        """
        Convert a hydrogen atom to deuterium (in-place).
        
        Args:
            atom: The atom to modify
        """
        if atom.element.name == "H":
            atom.name = atom.name.replace('H', 'D', 1)
            atom.element = gemmi.Element("D")

    def _is_labile_hydrogen(self, residue: gemmi.Residue) -> List[bool]:
        """
        Determine for each atom in a residue if it's a labile hydrogen.
        
        A hydrogen is labile if it's bonded to O, N, or S.
        This is approximated by checking if H follows an O/N/S atom.
        
        Args:
            residue: Residue to analyze
            
        Returns:
            List of booleans (one per atom in residue)
            True if the atom is a labile hydrogen
        """
        labile_list = []
        side_chain = False
        
        for atom in residue:
            if atom.element.name in ("O", "N", "S"):
                labile_list.append(False)
                side_chain = True
            else:
                if atom.element.name == "H":
                    labile_list.append(side_chain)
                else:
                    side_chain = False
                    labile_list.append(False)
        
        return labile_list

    def save(self, output_path: str) -> None:
        """
        Save the modified structure to a PDB file.
        
        Args:
            output_path: Path to the output file
            
        Raises:
            IOError: If writing fails
        """
        try:
            output_path = Path(output_path)
            self.structure.write_pdb(str(output_path))
            logger.info(f"Structure saved: {output_path}")
        except Exception as e:
            raise IOError(f"Error while saving: {e}")


# ============================================================================
#                           CONFIGURATION HANDLING
# ============================================================================

def load_pdb_config(config_file: str) -> dict:
    """
    Load deuteration configuration from an INI file.
    
    Expected format:
    [DEUTERATION]
    input_pdb = path/to/input.pdb
    output_pdb = path/to/output.pdb
    d2o_percent = 50
    
    [AMINO_ACIDS]
    ALA = true
    ARG = false
    ...
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dictionary with configuration parameters
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Read deuteration parameters
    input_pdb = config.get("DEUTERATION", "input_pdb")
    output_pdb = config.get("DEUTERATION", "output_pdb")
    d2o_percent = config.getfloat("DEUTERATION", "d2o_percent", fallback=0.0)
    
    # Read amino acid selection
    deuteration_vector = []
    for aa in AMINO_ACIDS:
        deuteration_vector.append(
            config.getboolean("AMINO_ACIDS", aa.code_3, fallback=False)
        )
    
    return {
        'input_pdb': input_pdb,
        'output_pdb': output_pdb,
        'd2o_percent': d2o_percent,
        'deuteration_vector': deuteration_vector
    }


# ============================================================================
#                           MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for standalone usage."""
    if len(sys.argv) < 4:
        print("Usage: python pdb_deuteration.py <input.pdb> <config.ini> <output.pdb>")
        print("\nAlternatively, use config.ini with paths specified inside:")
        print("  python pdb_deuteration.py <config.ini>")
        sys.exit(1)

    if len(sys.argv) == 2:
        # Config file only mode
        config_file = sys.argv[1]
        config = load_pdb_config(config_file)
        input_pdb = config['input_pdb']
        output_pdb = config['output_pdb']
        d2o_percent = config['d2o_percent']
        deuteration_vector = config['deuteration_vector']
    else:
        # Command line mode
        input_pdb = sys.argv[1]
        config_file = sys.argv[2]
        output_pdb = sys.argv[3]
        
        config = load_pdb_config(config_file)
        d2o_percent = config['d2o_percent']
        deuteration_vector = config['deuteration_vector']

    # Load and deuterate
    logger.info("=" * 60)
    logger.info("PDB DEUTERATION")
    logger.info("=" * 60)
    
    deuterator = PdbDeuteration(input_pdb)
    deuterator.apply_deuteration(deuteration_vector, d2o_percent)
    deuterator.save(output_pdb)
    
    logger.info("=" * 60)
    logger.info("Deuteration completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
