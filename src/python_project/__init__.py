"""
Module de génération de population pour l'optimisation de deutération
=====================================================================

Ce module gère la création et l'évolution de populations de chromosomes
représentant des configurations de deutération d'acides aminés.

Architecture:
    - AminoAcid: Dataclass représentant un acide aminé
    - Chromosome: Classe représentant une solution (configuration de deutération)
    - PopulationGenerator: Classe principale gérant la génération/évolution

Usage:
    # Première génération
    generator = PopulationGenerator(
        aa_list=AMINO_ACIDS,
        modifiable=restrictions,
        population_size=300, #Doit etre un multiple de 3
        d2o_initial=50,
        elitism=5
    )
    population = generator.generate_initial_population()55

    # Générations suivantes
    new_population = generator.generate_next_generation(
        previous_population=population,
        fitness_scores=[0.8, 0.7, ...],
        mutation_rate=0.15,
        crossover_rate=0.8,
        d2o_variation_rate=5
    )
"""

import random
import argparse
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

# ============================================================================
#                           PARSING DES ARGUMENTS
# ============================================================================

def parse_arguments():
    """
    Parse les arguments de ligne de commande pour configurer l'algorithme génétique.

    Returns:
        argparse.Namespace: Arguments parsés
    """
    parser = argparse.ArgumentParser(
        description="Algorithme génétique pour l'optimisation de la deutération d'acides aminés",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python __init__.py --population_size 50 --d2o_initial 60
  python __init__.py --mutation_rate 0.2 --crossover_rate 0.9 --elitism 5
  python __init__.py -p 100 -d 50 -e 3 -m 0.15 -c 0.8 -v 10
        """
    )

    # Paramètres de population
    parser.add_argument(
        '-p', '--population_size',
        type=int,
        default=30,
        help='Taille de la population (nombre de chromosomes, DOIT être multiple de 3). Défaut: 30'
    )

    parser.add_argument(
        '-e', '--elitism',
        type=int,
        default=2,
        help='Nombre d\'individus élites préservés à chaque génération (doit être ≤ population_size/3). Défaut: 5'
    )

    # Paramètres D2O
    #parser.add_argument(
    #    '-d', '--d2o_initial',
    #    type=int,
    #    default=50,
    #    help='Pourcentage initial de D2O (0-100). Utilisé pour la génération 0. Défaut: 50'
    #)

    parser.add_argument(
        '-v', '--d2o_variation_rate',
        type=int,
        default=5,
        help='Amplitude maximale de variation du D2O. Défaut: 50'
    )

    # Paramètres génétiques
    parser.add_argument(
        '-m', '--mutation_rate',
        type=float,
        default=0.15,
        help='Taux de mutation (0.0-1.0). Probabilité qu\'un gène soit muté. Défaut: 0.15'
    )

    parser.add_argument(
        '-c', '--crossover_rate',
        type=float,
        default=0.8,
        help='Taux de croisement (0.0-1.0). Probabilité de croisement entre deux parents. Défaut: 0.8'
    )

    # Paramètres d'exécution
    parser.add_argument(
        '-g', '--generations',
        type=int,
        default=1,
        help='Nombre de générations à exécuter. Défaut: 1'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Graine aléatoire pour la reproductibilité. Défaut: None (aléatoire)'
    )

    args = parser.parse_args()

    # Validation des arguments
    if args.population_size <= 0:
        parser.error("population_size doit être > 0")

    if args.population_size % 3 != 0:
        parser.error(f"population_size doit être un multiple de 3 (actuellement: {args.population_size})")

    if args.elitism < 0:
        parser.error("elitism doit être >= 0")

    if args.elitism >= args.population_size:
        parser.error("elitism doit être < population_size")

    if args.elitism > args.population_size // 3:
        parser.error(f"elitism doit être ≤ population_size/3 (max: {args.population_size // 3})")

    if not (0 <= args.d2o_variation_rate <= 100):
        parser.error("d2o_variation_rate doit être entre 0 et 100")

    if not (0 <= args.mutation_rate <= 1):
        parser.error("mutation_rate doit être entre 0.0 et 1.0")

    if not (0 <= args.crossover_rate <= 1):
        parser.error("crossover_rate doit être entre 0.0 et 1.0")

    if args.generations < 1:
        parser.error("generations doit être >= 1")

    return args

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
                 modifiable: List[bool]):
        """
        Initialise un chromosome

        Args:
            aa_list: Liste des acides aminés
            modifiable: Liste indiquant quels AA peuvent être modifiés
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

        self.d2o = 0

    def randomize_deuteration(self):
        """
        Initialise aléatoirement le vecteur de deutération (respectant les restrictions)
        """
        for i in range(self.n_aa):
            if self.modifiable[i]:
                self.deuteration[i] = random.choice([True, False])
            else:
                self.deuteration[i] = False

    def randomize_d2o(self):
        """
        Génère un pourcentage de D2O aléatoire entre 0 et 100 (distribution entiere)
        """
        self.d2o = random.randint(0, 100)

    def modify_d2o(self, variation_rate: int):
        """
        Permet une variation aleatoire du % de D2O
        Args:
             variation_rate: Amplitude de variation possible (5 correspond à ±5%)
        """
        variation = random.randint(-variation_rate, variation_rate)
        if 0 < self.d2o + variation < 100:
            self.d2o = self.d2o + variation
        elif 0 > self.d2o + variation :
            self.d2o = 0
        elif 100 < self.d2o + variation :
            self.d2o = 100

    def get_deuteration_count(self) -> int:
        """Compte le nombre d'AA deutérés"""
        return sum(self.deuteration)

    def copy(self) -> 'Chromosome':
        """Crée une copie du chromosome"""
        new_chrom = Chromosome(self.aa_list, self.modifiable)
        new_chrom.deuteration = self.deuteration.copy()
        new_chrom.d2o = self.d2o
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
        return " | ".join(result) + f" | D2O={self.d2o}% | fitness={self.fitness:.2f}"

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
                 #d2o_initial: int = 50,
                 elitism: int = 2,
                 d2o_variation_rate: int = 5):
        """Initialise le générateur de population."""
        self.aa_list = aa_list
        self.modifiable = modifiable
        self.population_size = population_size
        #self.d2o_initial = d2o_initial
        self.elitism = elitism
        self.d2o_variation_rate = d2o_variation_rate

        # Validation
        assert population_size % 3 == 0, f"population_size DOIT être un multiple de 3 (actuellement: {population_size})"

        assert population_size > 0, "population_size doit être > 0"
        #assert 0 <= d2o_initial <= 100, "d2o_initial doit être entre 0 et 100"
        assert elitism >= 0, "elitism doit être >= 0"
        assert elitism <= population_size // 3, f"elitism doit être ≤ population_size/3 (max: {population_size // 3})"
        assert d2o_variation_rate >= 0, "d2o_variation_rate doit être >= 0"

    def generate_initial_population(self) -> List[Chromosome]:
        """
        Génère la population initiale (génération 0).

        Crée une population de chromosomes avec deutération aléatoire
        et pourcentage de D2O aleatoire.

        Returns:
            Liste de chromosomes initialisés aléatoirement
        """
        population = []

        while len(population) < self.population_size:
            # Créer un chromosome
            chrom = Chromosome(self.aa_list, self.modifiable)

            # Initialiser aléatoirement le %D2O et la deutération
            chrom.randomize_d2o()
            chrom.randomize_deuteration()

            if self._unique_check(chrom, population):
                population.append(chrom)

        return population
    def generate_next_generation(self,
                                previous_population: List[Chromosome],
                                mutation_rate: float = 0.15,
                                crossover_rate: float = 0.7,
                                d2o_variation_rate: int = 5) -> List[Chromosome]:
        """
        Génère la prochaine génération selon la RÈGLE DES 3 TIERS:

        1. TIER 1 (n/3): Sélection
           - e meilleurs (élitisme)
           - (n/3 - e) par sélection probabiliste

        2. TIER 2 (n/3): Mutation
           - Chromosomes mutés à partir des sélectionnés

        3. TIER 3 (n/3): Crossover
           - Chromosomes issus de croisement des sélectionnés

        Args:
            previous_population: Population de la génération précédente
            mutation_rate: (à revoir) Probabilité de mutation par gène (0-1)
            crossover_rate: (à revoir) Probabilité de croisement (0-1)
            d2o_variation_rate: Amplitude max de variation de D2O (ex: 5%)

        Returns:
            Nouvelle population de n chromosomes
        """
        # Validation des entrées
        assert len(previous_population) == len(fitness_scores), "Le nombre de fitness doit correspondre à la taille de la population"
        assert len(previous_population) % 3 == 0, f"La taille de population doit être multiple de 3 (actuellement: {len(previous_population)})"
        assert 0 <= mutation_rate <= 1, "mutation_rate doit être entre 0 et 1"
        assert 0 <= crossover_rate <= 1, "crossover_rate doit être entre 0 et 1"


        # TIER 1 : Sélection (n/3 chromosomes)
        tier1_selectionnes = self._selection_tier1(previous_population)

        # TIER 2 : Mutation (n/3 chromosomes)
        tier2_mutes = self._mutation_tier2(tier1_selectionnes, d2o_variation_rate)

        # TIER 3 : Crossover (n/3 chromosomes)
        tier3_crossovers = self._crossover_tier3(tier1_selectionnes)

        # Assemblage de la nouvelle génération
        new_population = tier1_selectionnes + tier2_mutes + tier3_crossovers

        # Vérification de la taille
        assert len(new_population) == len(previous_population), f"Erreur: nouvelle population de taille {len(new_population)} au lieu de {len(previous_population)}"

        return new_population

    def _unique_check(self,new_chromosome: Chromosome,subpopulation: List[Chromosome]) -> bool:
        """
        Vérification que le chromosome qui va etre ajouter a la pop n'y est pas déjà
        :param new_chromosome: chromosome qu'on veut ajouter
        :param subpopulation: population avec des chromosomes uniques
        :return: False si un chromosome avec les memes motif de deuterisation et do2 equivalent (±5%) est trouvée, True sinon
        """
        for chromosome in subpopulation:
            if chromosome.deuteration == new_chromosome.deuteration and new_chromosome.d2o - 5 <= chromosome.d2o <= new_chromosome.d2o + 5 :
                print("2 chromosome IDENTIQUES")
                return False
        return True


    def _calculer_probabilites_selection(self, population: List[Chromosome]) -> List[float]:
        """
        Calcule les probabilités de sélection proportionnelles au fitness.

        CONTRAINTE CRITIQUE: JAMAIS 0% ni 100%

        Méthode: Softmax avec température pour garantir distribution > 0
        """
        fitness_array = np.array([chrom.fitness for chrom in population])

        # Éviter les valeurs négatives
        fitness_array = fitness_array - fitness_array.min() + 1e-10

        # Appliquer softmax avec température pour éviter 0% et 100%
        temperature = 0.5
        exp_fitness = np.exp(fitness_array / temperature)
        probas = exp_fitness / exp_fitness.sum()

        # Garantir min > 0 et max < 1
        epsilon = 1e-6
        probas = np.clip(probas, epsilon, 1.0 - epsilon)

        # Renormaliser
        probas = probas / probas.sum()

        return probas.tolist()

    def _selection_tier1(self, sorted_population: List[Chromosome]) -> List[Chromosome]:
        """
        TIER 1: Sélection de n/3 chromosomes

        Composition:
        - e meilleurs (élitisme)
        - (n/3 - e) par sélection probabiliste (fitness proportionnel)

        IMPORTANT: Probabilités JAMAIS 0% ni 100%
        """
        tier_size = self.population_size // 3
        selectionnes = []
        # Partie A: Élitisme - prendre les e meilleurs
        for i in range(self.elitism):
            selectionnes.append(sorted_population[i].copy())
            #xprint(f"Le chromosome selectionne a un d2o de {sorted_population[i].d2o} et une fitness de {sorted_population[i].fitness:.2f}")
        # Partie B: Sélection probabiliste pour le reste
        nombre_a_selectionner = tier_size - self.elitism
        #print (f"le nombre à selectionner est { nombre_a_selectionner }")

        if nombre_a_selectionner > 0:
            pop_a_selectionee = sorted_population[self.elitism:]
            # Calculer les probabilités proportionnelles au fitness
            probas = self._calculer_probabilites_selection(pop_a_selectionee)
            #print(probas)
            #print(f"le premier de la liste a un d2o de {pop_a_selectionee[i].d2o} et une fitness de {pop_a_selectionee[i].fitness:.2f}")
            # Sélectionner avec probabilité proportionnelle
            for _ in range(nombre_a_selectionner):
                selected = random.choices(pop_a_selectionee, weights=probas, k=1)[0]
                #print( f"les selectione on un d2o de {selected.d2o} et une fitness de {selected.fitness:.2f}")
                selectionnes.append(selected.copy())

        return selectionnes

    def _mutation_tier2(self,
                       selectionnes: List[Chromosome],
                       d2o_variation_rate: float) -> List[Chromosome]:
        """
        TIER 2: Génère n/3 chromosomes par mutation

        Pour chaque chromosome à générer:
        1. Choisir un parent au hasard parmi les sélectionnés
        2. Muter 1, 2 ou 3 AA aléatoirement
        3. 50% de chance: muter le D2O (variation aléatoire)
        """
        tier_size = self.population_size // 3
        mutes = []

        while len(mutes) < tier_size:
            # 1. Choisir un parent au hasard
            parent = random.choice(selectionnes)
            enfant = parent.copy()

            # 2. Muter les AA (1, 2 ou 3 mutations)
            nombre_mutations = random.choice([1, 2, 3])

            # Obtenir les indices modifiables
            indices_modifiables = [i for i in range(len(self.aa_list)) if self.modifiable[i]]

            # Sélectionner aléatoirement les AA à muter
            if len(indices_modifiables) >= nombre_mutations:
                indices_a_muter = random.sample(indices_modifiables, nombre_mutations)

                for idx in indices_a_muter:
                    enfant.deuteration[idx] = not enfant.deuteration[idx]

            # 3. Muter le D2O (50% de chance)
            if random.choice([True, False]):
                enfant.modify_d2o(d2o_variation_rate)
            if self._unique_check(enfant, mutes + selectionnes):
                mutes.append(enfant)

        return mutes

    def _crossover_tier3(self, selectionnes: List[Chromosome]) -> List[Chromosome]:
        """
        TIER 3: Génère n/3 chromosomes par crossover

        Pour chaque chromosome à générer:
        1. Choisir 2 parents au hasard
        2. Point de coupe aléatoire entre 1 et 19
        3. Créer enfant par crossover
        4. Transmission du D2O selon règles spéciales
        """
        tier_size = self.population_size // 3
        crossovers = []

        while len(crossovers) < tier_size:
            # 1. Choisir 2 parents au hasard
            parent1, parent2 = random.sample(selectionnes, 2)

            # 2. Point de coupe aléatoire (entre 1 et 19)
            point_coupe = random.randint(1, len(self.aa_list) - 1)

            # 3. Créer l'enfant par crossover
            enfant = Chromosome(self.aa_list, self.modifiable)

            # Crossover du vecteur de deutération
            enfant.deuteration = (
                parent1.deuteration[:point_coupe] +
                parent2.deuteration[point_coupe:]
            )

            # 4. Transmission du D2O selon règles
            if point_coupe == len(self.aa_list) // 2:  # Exactement à la moitié (position 10 pour 20 AA)
                # Transmetre un des 2 parent de maniere aleatoire
                enfant.d2o = random.choice([parent1.d2o, parent2.d2o])
            else:
                # Transmettre avec le parent du plus petit morceau
                if point_coupe < len(self.aa_list) // 2:
                    enfant.d2o = parent1.d2o
                else:
                    enfant.d2o = parent2.d2o

            crossovers.append(enfant)

        return crossovers



# Exemple d'utilisation
if __name__ == "__main__":

    # Parser les arguments de ligne de commande
    args = parse_arguments()

    # Fixer la graine aléatoire si spécifiée (pour reproductibilité)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Graine aléatoire fixée à: {args.seed}")

    # Afficher la configuration

    print("CONFIGURATION DE L'ALGORITHME GÉNÉTIQUE")
    print(f"Population size      : {args.population_size}")
    print(f"Élitisme             : {args.elitism}")
    #print(f"D2O initial          : {args.d2o_initial}%")
    print(f"D2O variation rate   : ±{args.d2o_variation_rate}%")
    print(f"Mutation rate        : {args.mutation_rate}")
    print(f"Crossover rate       : {args.crossover_rate}")
    print(f"Générations          : {args.generations}")


    # Créer et exécuter l'algorithme génétique
    print("\n>>> GÉNÉRATION 0 - Création de la population")
    generator = PopulationGenerator(
        aa_list=AMINO_ACIDS,
        modifiable=restrictions,
        population_size=args.population_size,
        #d2o_initial=args.d2o_initial,
        elitism=args.elitism,
        d2o_variation_rate=args.d2o_variation_rate,
    )

    #Génération et affichage de la population initiale
    pop = generator.generate_initial_population()

    # Simuler des scores de fitness aléatoires (normalement calculés par SANS)
    print("\n>>> Simulation de l'évaluation par SANS...")
    fitness_scores = [random.uniform(0.3, 0.9) for _ in pop]

    #atribué la fitness au chromosomes
    for chrom, fitness in zip(pop, fitness_scores):
        chrom.fitness = fitness

    sorted_pop = sorted(pop,
                               key=lambda x: x.fitness,
                               reverse=True)

    for i, chromosome in enumerate(sorted_pop):
        print(f"{i+1}. {chromosome}")


    # Génération n : Évolution
    for gen in range(1, args.generations + 1):
        print(f"\n>>> GÉNÉRATION {gen} - Évolution de la population")
        pop = generator.generate_next_generation(
            previous_population=sorted_pop,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            d2o_variation_rate=args.d2o_variation_rate
        )

        # Simuler des scores de fitness aléatoires (normalement calculés par SANS)
        print("\n>>> Simulation de l'évaluation par SANS...")
        fitness_scores = [random.uniform(0.3, 0.9) for _ in pop]

        # atribué la fitness au chromosomes
        for chrom, fitness in zip(pop, fitness_scores):
            chrom.fitness = fitness

        sorted_pop = sorted(pop,
                              key=lambda x: x.fitness,
                              reverse=True)

        for i, chromosome in enumerate(sorted_pop):
            print(f"{i+1}. {chromosome}")
