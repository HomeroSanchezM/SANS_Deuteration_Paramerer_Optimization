import sys
import gemmi
from dataclasses import dataclass
import random
from pathlib import Path
from __init__ import *
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PdbDeuteration:
    """
    Main class for deuteration of the PDB file.
    """
    def __init__(self, pdb_file: str):

        self.pdb_path = Path(pdb_file)

        if not self.pdb_path.exists():
            raise FileNotFoundError(f"Fichier PDB introuvable: {pdb_path}")
        try:
            self.structure = gemmi.read_structure(str(self.pdb_path))
            logger.info(f"Structure chargée: {self.pdb_path.name}")
            logger.info(f"Modèles: {len(self.structure)}, "
                       f"Chaînes: {sum(len(model) for model in self.structure)}")
        except Exception as e:
            raise RuntimeError(f"Erreur lors du parsing PDB: {e}")

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <fichier.pdb> [output.pdb]")
        sys.exit(1)

    pdb_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output_deuterated.pdb"

    test_deterator = PdbDeuteration(pdb_file)

    # Ala(D) | Arg(H) | Asn(D) | Asp(D) | Cys(H) | Glu(H) | Gln(H) | Gly(D) | His(H) | Ile(D) | Leu(D) | Lys(H) | Met(D) | Phe(D) | Pro(H) | Ser(D) | Thr(D) | Trp(H) | Tyr(D) | Val(H) | D2O=47% |
    test_deuteration_vector = [True, False, True, True, False, False, False, True, False, True, True, False, True, True,
                               False, True, True, False, True, False]

    test_chromosome = Chromosome(aa_list=AMINO_ACIDS,
                                 modifiable=restrictions)
    test_chromosome.deuteration = test_deuteration_vector
    test_chromosome.d2o = 0.47
