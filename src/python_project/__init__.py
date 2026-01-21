import random

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


# ============================================================================
#                   DÉFINITION DES STRUCTURES DE DONNÉES
# ============================================================================



@dataclass
class AminoAcid:
    """
    Représente un acide aminé avec ses codes nomenclature.

    Attributes:
        name (str): Nom complet de l'acide aminé (ex: "Alanine")
        code_3 (str): Code à 3 lettres (ex: "Ala")
        code_1 (str): Code à 1 lettre (ex: "A")
    """
    name: str
    code_3: str
    code_1: str


# Liste des acides aminés communs
AMINO_ACIDS = [
    AminoAcid("Alanine", "Ala", "A" ),
    AminoAcid("Arginine", "Arg", "R" ),
    AminoAcid("Asparagine", "Asn", "N" ),
    AminoAcid("Acide aspartique", "Asp", "D" ),
    AminoAcid("Cystéine", "Cys", "C"),
    AminoAcid("Acide glutamique", "Glu", "E" ),
    AminoAcid("Glutamine", "Gln", "Q"),
    AminoAcid("Glycine", "Gly", "G" ),
    AminoAcid("Histidine", "His", "H"),
    AminoAcid("Isoleucine", "Ile", "I" ),
    AminoAcid("Leucine", "Leu", "L" ),
    AminoAcid("Lysine", "Lys", "K" ),
    AminoAcid("Méthionine", "Met", "M"),
    AminoAcid("Phénylalanine", "Phe", "F"),
    AminoAcid("Proline", "Pro", "P"),
    AminoAcid("Sérine", "Ser", "S"),
    AminoAcid("Thréonine", "Thr", "T"),
    AminoAcid("Tryptophane", "Trp", "W"),
    AminoAcid("Tyrosine", "Tyr", "Y"),
    AminoAcid("Valine", "Val", "V")
]

#Liste pour déterminer si un AA est deutérable (True) ou non (False)
restrictions = [
        True,   # Ala
        False,  # Arg
        True,   # Asn
        True,   # Asp
        False,  # Cys
        False,  # Glu
        True,   # Gln
        True,   # Gly
        False,  # His
        True,   # Ile
        True,   # Leu
        True,   # Lys
        True,   # Met
        True,   # Phe
        True,   # Pro
        True,   # Ser
        True,   # Thr
        True,   # Trp
        True,   # Typ
        True,   # Val
    ]

# ============================================================================
#                           CLASSE CHROMOSOME
# ============================================================================



class Chromosome:
    """
    Représente un chromosome dans l'algorithme génétique

    Un chromosome encode:
        - Quels acides aminés sont deutérés (vecteur booléen)
        - Le pourcentage de D2O utilisé (int entre 0 et 100)
        - Le score de fitness (calculé à partir des données SANS)

    Attributes:
        aa_list (List[AminoAcid]): Liste des acides aminés
        modifiable (List[bool]): Restrictions de deutération par AA
        deuteration (List[bool]): Vecteur de deutération (True = deutéré)
        d2o (int): Pourcentage de D2O (0-100)
        fitness (float): Score de fitness du chromosome

    """

    def __init__(self,
                 aa_list: List[AminoAcid],
                 modifiable: List[bool],
                 d2o_initial: int = 50):
        """
        Initialise un chromosome

        Args:
            aa_list: Liste des acides aminés
            modifiable: Liste indiquant quels AA peuvent être modifiés
            d2o_initial: Pourcentage initial de D2O (défaut: 50)
        """
        self.aa_list = aa_list
        self.n_aa = len(aa_list)

        # Si pas de restrictions spécifiées, tous sont modifiables
        if modifiable is None:
            self.modifiable = [True] * self.n_aa
        else:
            assert len(modifiable) == self.n_aa, "modifiable doit avoir la même taille que aa_list, il doit avoir {self.n_aa} éléments"
            self.modifiable = modifiable

        # Vecteur de deutération (True = deutéré, False = non deutéré)
        self.deuteration = [False] * self.n_aa

        # Fitness du chromosome
        self.fitness = 0.0

        self.d2o = self._validate_d2o(d2o_initial)

    @staticmethod
    def _validate_d2o(value: int) -> int:
        """Valide les borne de D2O entre 0 et 100"""
        return max(0, min(int(value), 100))

    def randomize_deuteration(self):
        """
        Initialise aléatoirement le vecteur de deutération (respectant les restrictions)
        """
        for i in range(self.n_aa):
            if self.modifiable[i]:
                self.deuteration[i] = random.choice([True, False])
            else:
                self.deuteration[i] = False

    def modify_d20(self, variation_rate: int):
        """
        Permet une variation aleatoire du % de D2O
        Args:
             variation_rate: Amplitude de variation possible (5 correspond à ±5%)
        """
        variation = random.randint(-variation_rate, variation_rate)
        self.d2o = self._validate_d2o(self.d2o + variation)


    def get_deuteration_count(self) -> int:
        """Compte le nombre d'AA deutérés"""
        return sum(self.deuteration)

    def copy(self) -> 'Chromosome':
        """Crée une copie du chromosome"""
        new_chrom = Chromosome(self.aa_list, self.modifiable, self.d2o)
        new_chrom.deuteration = self.deuteration.copy()
        new_chrom.fitness = self.fitness
        return new_chrom

    def __str__(self) -> str:
        """Représentation textuelle du chromosome"""
        result = []
        for i, aa in enumerate(self.aa_list):
            if self.deuteration[i]:
                result.append(f"{aa.code_3}(D)")
            else:
                result.append(f"{aa.code_3}(H)")
        return " | ".join(result) + f" | D2O={self.d2o}% | fitness={self.fitness}"

    def to_dict(self) -> dict:
        """Convertit le chromosome en dictionnaire pour sérialisation."""
        return {
            'deuteration': self.deuteration.copy(),
            'd2o': self.d2o,
            'fitness': self.fitness,
            'deuteration_count': self.get_deuteration_count()
        }


# ============================================================================
#                       CLASSE POPULATION GENERATOR
# ============================================================================

class PopulationGenerator:
    """
    Générateur de populations pour l'algorithme génétique.

    Gère la création de populations initiales et l'évolution de populations
    existantes via sélection, croisement et mutation.

    Attributes:
        aa_list: Liste des acides aminés
        modifiable: Restrictions de deutération
        population_size: Taille de la population
        d2o_initial: Pourcentage initial de D2O
        elitism: Nombre d'élites préservés entre générations
    """
    def __init__(self,
                 aa_list: List[AminoAcid],
                 modifiable: List[bool],
                 population_size: int = 100,
                 d2o_initial: int = 50,
                 elitism: int = 2,
                 d2o_variation_rate: int = 5):
        """Initialise le générateur de population."""
        self.aa_list = aa_list
        self.modifiable = modifiable
        self.population_size = population_size
        self.d2o_initial = d2o_initial
        self.elitism = elitism
        self.d2o_variation_rate = d2o_variation_rate

        # Validation
        assert population_size > 0, "population_size doit être > 0"
        assert 0 <= d2o_initial <= 100, "d2o_initial doit être entre 0 et 100"
        assert elitism >= 0, "elitism doit être >= 0"
        assert elitism < population_size, "elitism doit être < population_size"
        assert 0 <= d2o_variation_rate <= 100, "d20_vatiation_rate doit être entre 0 et 100"

    def generate_initial_population(self) -> List[Chromosome]:
        """
        Génère la population initiale (génération 0).

        Crée une population de chromosomes avec deutération aléatoire
        et pourcentage de D2O variant autour d'un D2O initial.

        Returns:
            Liste de chromosomes initialisés aléatoirement
        """
        population = []

        for _ in range(self.population_size):
            # Créer un chromosome
            chrom = Chromosome(self.aa_list, self.modifiable, self.d2o_initial)

            # Initialiser aléatoirement la deutération
            chrom.randomize_deuteration()

            # Initialiser les variation de D2O autour de d20_initial
            chrom.modify_d20(self.d2o_variation_rate)

            population.append(chrom)

        return population



# Exemple d'utilisation
if __name__ == "__main__":

    # Créer et exécuter l'algorithme génétique
    generator = PopulationGenerator(
        aa_list= AMINO_ACIDS,
        modifiable= restrictions,
        population_size= 10,
        d2o_initial=50,
        elitism= 2,
        d2o_variation_rate= 5,
    )

    #Génération et affichage de la population initiale
    pop_0 = generator.generate_initial_population()

    for chromosome in pop_0:
        print(chromosome)