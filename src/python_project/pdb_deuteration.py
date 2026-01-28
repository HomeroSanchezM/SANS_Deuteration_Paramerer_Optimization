import sys
import gemmi
from dataclasses import dataclass
import random
from __init__ import *

if __name__ == "__main__":
    # Ala(D) | Arg(H) | Asn(D) | Asp(D) | Cys(H) | Glu(H) | Gln(H) | Gly(D) | His(H) | Ile(D) | Leu(D) | Lys(H) | Met(D) | Phe(D) | Pro(H) | Ser(D) | Thr(D) | Trp(H) | Tyr(D) | Val(H) | D2O=47% |
    test_deuteration_vector = [True, False, True, True, False, False, False, True, False, True, True, False, True, True,
                               False, True, True, False, True, False]

    test_chromosome = Chromosome(aa_list=AMINO_ACIDS,
                                 modifiable=restrictions)
    test_chromosome.deuteration = test_deuteration_vector
    test_chromosome.d2o = 0.47
