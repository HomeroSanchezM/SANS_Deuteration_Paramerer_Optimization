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
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        try:
            self.structure = gemmi.read_structure(str(self.pdb_path))
            logger.info(f"Structure loaded: {self.pdb_path.name}")
            logger.info(f"Models: {len(self.structure)}, "
                       f"Chains: {sum(len(model) for model in self.structure)}")
        except Exception as e:
            raise RuntimeError(f"Error while parsing PDB: {e}")

        # Stats for validation
        self.stats = {
            'total_atoms': 0,
            'hydrogen_atoms': 0,
            'deuterium_atoms': 0,
            'labile_H': 0,
            'non_labile_H': 0
        }
        self._count_atoms()

    def _count_atoms(self) -> None:
        """Count all, H and D atoms for stats."""
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
                          chromosome : Chromosome) -> None:
        """
        Apply the deuteration of the PDB file.
        :param deuteration_vector: List of booleans representing deuterations
        :param d2o_percent: percent of deuterations of labile H
        """
        # Validation des entrées
        if len(chromosome.deuteration) != 20:
            raise ValueError(f"aa_vector have to contain 20 elements, {len(aa_vector)} recieved")

        if not 0 <= chromosome.d2o <= 100:
            raise ValueError(f"d2o_percent have to be between 0 and 100, {d2o_percent} recieved")

        logger.info(f"Apply deuteration : D₂O = {chromosome.d2o:.2f}%")
        logger.info(f"Deutered AA: {sum(chromosome.deuteration)}/20")

        # Parcours de la structure
        for model in self.structure:
            for chain in model:
                for residue in chain:

                    residue_name = residue.name.strip().upper()

                    # Vérifier si c'est un AA standard
                    if residue_name not in AA_DICT:
                        logger.debug(f"Résidu non-standard ignoré: {residue_name}")
                        continue

                    labile_vector = self._is_labile_hydrogen(residue)
                    print(labile_vector)
                    aa_index = AA_DICT[residue_name]
                    should_deuterate = chromosome.deuteration[aa_index]
                    if should_deuterate:
                        print(f"=========Le residue {residue_name} va etre deuteré========= ")
                    else:
                        print(f"=========Les residue {residue_name} va pas etre deuteré=======")


                    # Traitement de chaque atome du résidu
                    for atom, is_labile in zip(residue, labile_vector):
                        #print(f"atom element name: {atom.element.name}") #que H (derniere colonne)
                        print(f"atom name: {atom.name}") # H1 (deuxieme colonne)
                        element = atom.element.name

                        if is_labile:
                            # H labile: appliquer selon D₂O%
                            if random.random() * 100 < chromosome.d2o:
                                if element == "H":
                                    print("L'atome labile va etre deuteré")
                                    self._convert_atom_H_to_D(atom)
                                    print(f"New atom name: {atom.name}")
                        element = atom.element.name
                        # Ignorer les atomes qui ne sont ni H ni D
                        #if element not in ("H"):
                        #    continue

                        #print(f"atom element name: {atom.element.name}")  # que H (derniere colonne)
                        #print(f"atom name: {atom.name}")  # H1 (deuxieme colonne)

                        if should_deuterate:
                            if element == "H":
                                print("L'atome dans le AA select va etre deutéré")
                                self._convert_atom_H_to_D(atom)
                                print(f"New atom name: {atom.name}")

                        #print(f"atom element name: {atom.element.name}")  # que H (derniere colonne)
                        #print(f"atom name: {atom.name}")  # H1 (deuxieme colonne)

    def _convert_atom_H_to_D(self, atom: gemmi.Atom) -> None:
        """
        Convertit un atome H en D (in-place).

        Args:
            atom: L'atome à modifier
        """
        if atom.element.name == "H":
            atom.name = atom.name.replace('H', 'D', 1)
            # Mettre à jour l'élément en utilisant gemmi.Element
            atom.element = gemmi.Element("D")

    def _is_labile_hydrogen(self, residue: gemmi.Residue) -> List[bool]:
        """
        Find for eatch atom of a residue if is a labile H or not
        H have to be linked to O, N or S
        :param residue: Residue to test
        :return: a list of bool, True if the atom is labile
        """
        list_labile = []
        #side_chain = True
        for atom in residue:
            #print(f"atom name: {atom.name}, element name {atom.element.name}")
            if atom.element.name in ("O", "N", "S"):
                #print("les atomes suivant peuvent etre Labiles")
                list_labile.append(False)
                side_chain = True
                #print("side chain devient True 1")
            else:
                if atom.element.name == "H":
                    if side_chain:
                        #print("comme on est dans une chaine O, N ou S le H est labile")
                        list_labile.append(True)
                    else:
                        #print("comme on est PAS dans une chaine, H est PAS labile")
                        list_labile.append(False)
                else:
                    #print("les atomes ne peuvent PAS etre Labiles")
                    side_chain = False
                    list_labile.append(False)
        return list_labile

    def save(self, output_path: str) -> None:
        """
        Sauvegarde la structure modifiée.

        Args:
            output_path: Chemin du fichier de sortie

        Raises:
            IOError: Si l'écriture échoue
        """
        try:
            output_path = Path(output_path)
            self.structure.write_pdb(str(output_path))
            logger.info(f"Structure saved: {output_path}")
        except Exception as e:
            raise IOError(f"Error while saving: {e}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <fichier.pdb> [output.pdb]")
        sys.exit(1)

    pdb_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output_deuterated.pdb"

    test_deuterator = PdbDeuteration(pdb_file)

    #print(test_deuterator.stats)

    # Ala(D) | Arg(H) | Asn(D) | Asp(D) | Cys(H) | Glu(H) | Gln(H) | Gly(D) | His(H) | Ile(D) | Leu(D) | Lys(H) | Met(D) | Phe(D) | Pro(H) | Ser(D) | Thr(D) | Trp(H) | Tyr(D) | Val(H) | D2O=47% |
    test_deuteration_vector = [True, False, True, True, False, False, False, True, False, True, True, False, True, True,
                               False, True, True, False, True, False]

    test_chromosome = Chromosome(aa_list=AMINO_ACIDS,
                                 modifiable=restrictions)
    test_chromosome.deuteration = test_deuteration_vector
    test_chromosome.d2o = 47

    test_deuterator.apply_deuteration(test_chromosome)
    #save pdb
    test_deuterator.save(output_file)