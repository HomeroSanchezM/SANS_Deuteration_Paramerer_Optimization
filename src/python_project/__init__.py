"""
Population Generation Module for Deuteration Optimization
==========================================================

This module handles the creation and evolution of chromosome populations
representing amino acid deuteration configurations.

Architecture:
    - AminoAcid: Dataclass representing an amino acid
    - Chromosome: Class representing a solution (deuteration configuration)
    - PopulationGenerator: Main class managing generation/evolution

Usage:
    # First generation
    generator = PopulationGenerator(
        aa_list=AMINO_ACIDS,
        modifiable=restrictions,
        population_size=300, #Must be a multiple of 3
        d2o_initial=50,
        elitism=5
    )
    population = generator.generate_initial_population()55

    # Following generations
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

from create_pop_v2 import restrictions


# ============================================================================
#                           ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """
    Parses command-line arguments to configure the genetic algorithm.

    Returns:
        argparse.Namespace: Parsed arguments

    """
    parser = argparse.ArgumentParser(
        description="Genetic algorithm for amino acid deuteration optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage example:
  python __init__.py --population_size 50 --d2o_initial 60
  python __init__.py --mutation_rate 0.2 --crossover_rate 0.9 --elitism 5
  python __init__.py -p 100 -d 50 -e 3 -m 0.15 -c 0.8 -v 10
        """
    )

    # Population parameters
    parser.add_argument(
        '-p', '--population_size',
        type=int,
        default=30,
        help='Population size (number of chromosomes, MUST be a multiple of 3). Default: 30'
    )

    parser.add_argument(
        '-e', '--elitism',
        type=int,
        default=2,
        help='Number of elite individuals preserved at each generation (must be ≤ population_size/3). Default: 5'
    )

    # D2O parameters
    #parser.add_argument(
    #    '-d', '--d2o_initial',
    #    type=int,
    #    default=50,
    #    help='Initial D2O percentage (0-100). Used for generation 0. Default: 50'
    #)

    parser.add_argument(
        '-v', '--d2o_variation_rate',
        type=int,
        default=5,
        help='Maximum D2O variation amplitude. Default: 50'
    )

    # Genetic parameters
    parser.add_argument(
        '-m', '--mutation_rate',
        type=float,
        default=0.15,
        help='Mutation rate (0.0-1.0). Probability that a gene will be mutated. Default: 0.15'
    )

    parser.add_argument(
        '-c', '--crossover_rate',
        type=float,
        default=0.8,
        help='Crossover rate (0.0-1.0). Probability of crossover between two parents. Default: 0.8'
    )

    # Execution parameters
    parser.add_argument(
        '-g', '--generations',
        type=int,
        default=1,
        help='Number of generations to execute. Default: 1'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility. Default: None (random)'
    )

    args = parser.parse_args()

    # Argument validation
    if args.population_size <= 0:
        parser.error("population_size must be > 0")

    if args.population_size % 3 != 0:
        parser.error(f"population_size must be a multiple of 3 (currently: {args.population_size})")

    if args.elitism < 0:
        parser.error("elitism must be >= 0")

    if args.elitism >= args.population_size:
        parser.error("elitism must be < population_size")

    if args.elitism > args.population_size // 3:
        parser.error(f"elitism must be ≤ population_size/3 (max: {args.population_size // 3})")

    if not (0 <= args.d2o_variation_rate <= 100):
        parser.error("d2o_variation_rate must be between 0 and 100")

    if not (0 <= args.mutation_rate <= 1):
        parser.error("mutation_rate must be between 0.0 and 1.0")

    if not (0 <= args.crossover_rate <= 1):
        parser.error("crossover_rate must be between 0.0 and 1.0")

    if args.generations < 1:
        parser.error("generations must be >= 1")

    return args

# ============================================================================
#                   DATA STRUCTURE DEFINITIONS
# ============================================================================

@dataclass
class AminoAcid:
    """
    Represents an amino acid with its nomenclature codes.

    Attributes:
        name (str): Full name of the amino acid (e.g., "Alanine")
        code_3 (str): 3-letter code (e.g., "Ala")
        code_1 (str): 1-letter code (e.g., "A")
    """
    name: str
    code_3: str
    code_1: str

# List of common amino acids
AMINO_ACIDS = [
    AminoAcid("Alanine", "ALA", "A" ),
    AminoAcid("Arginine", "ARG", "R" ),
    AminoAcid("Asparagine", "ASN", "N" ),
    AminoAcid("Aspartic acid", "ASP", "D" ),
    AminoAcid("Cysteine", "CYS", "C"),
    AminoAcid("Glutamic acid", "GLU", "E" ),
    AminoAcid("Glutamine", "GLN", "Q"),
    AminoAcid("Glycine", "GLY", "G" ),
    AminoAcid("Histidine", "HIS", "H"),
    AminoAcid("Isoleucine", "ILE", "I" ),
    AminoAcid("Leucine", "LEU", "L" ),
    AminoAcid("Lysine", "LYS", "K" ),
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


#List to determine if an AA is deuterable (True) or not (False)
#restrictions = [
#        True,   # Ala
#        False,  # Arg
#        True,   # Asn
#        True,   # Asp
#        False,  # Cys
#        False,  # Glu
#        True,   # Gln
#        True,   # Gly
#        False,  # His
#        True,   # Ile
#        True,   # Leu
#        True,   # Lys
#        True,   # Met
#        True,   # Phe
#        True,   # Pro
#        True,   # Ser
#        True,   # Thr
#        True,   # Trp
#        True,   # Typ
#        True,   # Val
#    ]

restrictions = [True] *20
# ============================================================================
#                           CHROMOSOME CLASS
# ============================================================================

class Chromosome:
    """
    Represents a chromosome in the genetic algorithm

    A chromosome = a complete deuteration solution
    - 1 deuteration vector of length 20 (1 per amino acid)
    - 1 D2O value

    Attributes:
        aa_list (List[AminoAcid]): List of amino acids
        modifiable (List[bool]): Indicates which AA can be modified
        deuteration (List[bool]): Deuteration state for each AA
        d2o (int): D2O percentage (0-100)
        fitness (float): Fitness score (set by SANS evaluation)
    """

    def __init__(self, aa_list: List[AminoAcid], modifiable: List[bool]):
        """
        Initializes an empty chromosome.

        Args:
            aa_list: List of amino acids
            modifiable: Indicates which positions can be deuterated
        """
        self.aa_list = aa_list
        self.modifiable = modifiable
        self.deuteration = [False] * len(aa_list)  # All non-deuterated by default
        self.d2o = 50  # Default value
        self.fitness = 0.0  # Will be set after SANS evaluation

    def copy(self) -> 'Chromosome':
        """
        Creates a deep copy of the chromosome.

        Returns:
            Chromosome: Independent copy
        """
        nouveau = Chromosome(self.aa_list, self.modifiable)
        nouveau.deuteration = self.deuteration.copy()
        nouveau.d2o = self.d2o
        nouveau.fitness = self.fitness
        return nouveau

    def randomize_deuteration(self) -> None:
        """
        Randomly initializes the deuteration vector.

        Rules:
        - Respects modifiable constraints
        - Each modifiable AA has 50% chance to be deuterated
        """
        for i in range(len(self.aa_list)):
            if self.modifiable[i]:
                self.deuteration[i] = random.choice([True, False])

    def modify_d2o(self, variation_range: int = 10) -> None:
        """
        Modifies D2O value with random variation.

        Args:
            variation_range: Maximum variation amplitude

        Rules:
        - New value: current ± [0, variation_range]
        - Constrained to [0, 100]
        """
        variation = random.randint(-variation_range, variation_range)
        self.d2o = max(0, min(100, self.d2o + variation))

    def set_d2o(self, value: int) -> None:
        """
        Sets D2O value directly.

        Args:
            value: New D2O value (will be clamped to [0, 100])
        """
        self.d2o = max(0, min(100, value))

    def __str__(self) -> str:
        """
        Readable representation showing deuterated amino acids.

        Returns:
            str: "{AA codes} | D2O: X% | Fitness: Y"
        """
        deuterated_aa = [
            self.aa_list[i].code_3
            for i in range(len(self.aa_list))
            if self.deuteration[i]
        ]
        return f"{', '.join(deuterated_aa) if deuterated_aa else 'None'} | D2O: {self.d2o}% | Fitness: {self.fitness:.2f}"

    def __repr__(self) -> str:
        """Technical representation."""
        return f"Chromosome(d2o={self.d2o}, fitness={self.fitness:.2f})"

    def __eq__(self, other: 'Chromosome') -> bool:
        """
        Checks if two chromosomes are identical.

        Two chromosomes are identical if:
        - Same deuteration vector
        - Same D2O value

        Fitness is NOT included in comparison
        """
        if not isinstance(other, Chromosome):
            return False
        return (self.deuteration == other.deuteration and
                self.d2o == other.d2o)

    def __hash__(self) -> int:
        """
        Allows chromosomes to be used in sets/dicts.

        Hash based on deuteration + D2O (not fitness)
        """
        return hash((tuple(self.deuteration), self.d2o))


# ============================================================================
#                       POPULATION GENERATOR CLASS
# ============================================================================

class PopulationGenerator:
    """
    Manages the generation and evolution of chromosome populations.

    This class implements the complete genetic algorithm:
    - Initial generation 0: random population
    - Generation n: evolution via selection/mutation/crossover

    Main parameters:
        population_size: Must be a multiple of 3 (for 3-tier architecture)
        elitism: Number of best individuals preserved (must be ≤ population_size/3)
    """

    def __init__(self,
                 aa_list: List[AminoAcid],
                 modifiable: List[bool],
                 population_size: int = 30,
                 #d2o_initial: int = 50,
                 elitism: int = 5,
                 d2o_variation_rate: int = 5):
        """
        Initializes the generator.

        Args:
            aa_list: List of amino acids
            modifiable: Modification constraints
            population_size: Total population size (MUST be multiple of 3)
            #d2o_initial: Initial D2O percentage for generation 0
            elitism: Number of elite individuals (≤ population_size/3)

        Raises:
            ValueError: If population_size not multiple of 3
            ValueError: If elitism > population_size/3
        """
        if population_size % 3 != 0:
            raise ValueError(f"population_size must be a multiple of 3 (received: {population_size})")

        if elitism > population_size // 3:
            raise ValueError(f"elitism ({elitism}) must be ≤ population_size/3 ({population_size // 3})")

        self.aa_list = aa_list
        self.modifiable = modifiable
        self.population_size = population_size
        #self.d2o_initial = d2o_initial
        self.elitism = elitism
        self.d2o_variation_rate = d2o_variation_rate

    def generate_initial_population(self) -> List[Chromosome]:
        """
        Generates the initial population (Generation 0).

        Method:
        - Creates population_size unique chromosomes
        - Each one with random deuteration
        - All with D2O = d2o_initial

        Returns:
            List[Chromosome]: Initial population (not yet evaluated)
        """
        population = []
        tentatives = 0
        max_tentatives = self.population_size * 100  # Security to avoid infinite loop

        while len(population) < self.population_size and tentatives < max_tentatives:
            chrom = Chromosome(self.aa_list, self.modifiable)
            chrom.randomize_deuteration()
            chrom.set_d2o(random.randint(0, 100))

            # Only add if unique
            if chrom not in population:
                population.append(chrom)

            tentatives += 1

        if len(population) < self.population_size:
            raise RuntimeError(f"Could not generate {self.population_size} unique chromosomes")

        return population

    def generate_next_generation(self,
                                previous_population: List[Chromosome],
                                mutation_rate: float = 0.15,
                                crossover_rate: float = 0.8,
                                d2o_variation_rate: int = 5) -> List[Chromosome]:
        """
        Generates the next generation from the current population.

        ⚠️  IMPORTANT: previous_population MUST have fitness values set

        3-tier architecture:
        ┌──────────────────────────────────────────────────────────────┐
        │  TIER 1 - SELECTION (n/3 chromosomes)                       │
        │  - e best (elitism)                                          │
        │  - (n/3 - e) by probabilistic selection                     │
        │  → Serves as parents for tiers 2 and 3                      │
        └──────────────────────────────────────────────────────────────┘
        ┌──────────────────────────────────────────────────────────────┐
        │  TIER 2 - MUTATION (n/3 chromosomes)                        │
        │  - Random parent from tier 1                                 │
        │  - Mutate 1, 2, or 3 AA                                     │
        │  - 50% chance: mutate D2O                                    │
        └──────────────────────────────────────────────────────────────┘
        ┌──────────────────────────────────────────────────────────────┐
        │  TIER 3 - CROSSOVER (n/3 chromosomes)                       │
        │  - 2 random parents from tier 1                              │
        │  - Random cut point (1-19)                                   │
        │  - D2O transmission according to special rules               │
        └──────────────────────────────────────────────────────────────┘

        Args:
            previous_population: Population with fitness set (sorted by fitness)
            mutation_rate: Probability of mutation (currently unused)
            crossover_rate: Probability of crossover (currently unused)
            d2o_variation_rate: Maximum D2O variation

        Returns:
            List[Chromosome]: New population (fitness not yet set)
        """
        # Validation
        if len(previous_population) != self.population_size:
            raise ValueError(f"Previous population size ({len(previous_population)}) "
                           f"!= expected size ({self.population_size})")

        if not all(hasattr(c, 'fitness') for c in previous_population):
            raise ValueError("All chromosomes must have a fitness value")

        # Sort by decreasing fitness
        sorted_pop = sorted(previous_population,
                          key=lambda x: x.fitness,
                          reverse=True)

        # Generation of 3 tiers
        tier1 = self._selection_tier1(sorted_pop)
        tier2 = self._mutation_tier2(tier1, d2o_variation_rate)
        tier3 = self._crossover_tier3(tier1)

        # Combine the 3 tiers
        new_population = tier1 + tier2 + tier3

        # Final validation
        if len(new_population) != self.population_size:
            raise RuntimeError(f"Error: new population = {len(new_population)}, "
                             f"expected {self.population_size}")

        return new_population

    def _unique_check(self, chromosome: Chromosome, population: List[Chromosome]) -> bool:
        """
        Checks if a chromosome is unique in the population.

        Args:
            chromosome: Chromosome to check
            population: Existing population

        Returns:
            bool: True if unique, False otherwise
        """
        for existing in population:
            if chromosome == existing:
                return False
        return True


    def _calculer_probabilites_selection(self, population: List[Chromosome]) -> List[float]:
        """
        Calculates selection probabilities proportional to fitness.

        CRITICAL CONSTRAINT: NEVER 0% nor 100%

        Method: Softmax with temperature to guarantee distribution > 0
        """
        fitness_array = np.array([chrom.fitness for chrom in population])

        # Avoid negative values
        fitness_array = fitness_array - fitness_array.min() + 1e-10

        # Apply softmax with temperature to avoid 0% and 100%
        temperature = 0.5
        exp_fitness = np.exp(fitness_array / temperature)
        probas = exp_fitness / exp_fitness.sum()

        # Guarantee min > 0 and max < 1
        epsilon = 1e-6
        probas = np.clip(probas, epsilon, 1.0 - epsilon)

        # Renormalize
        probas = probas / probas.sum()

        return probas.tolist()

    def _selection_tier1(self, sorted_population: List[Chromosome]) -> List[Chromosome]:
        """
        TIER 1: Selection of n/3 chromosomes

        Composition:
        - e best (elitism)
        - (n/3 - e) by probabilistic selection (fitness proportional)

        IMPORTANT: Probabilities NEVER 0% nor 100%
        """
        tier_size = self.population_size // 3
        selectionnes = []
        # Part A: Elitism - take the e best
        for i in range(self.elitism):
            selectionnes.append(sorted_population[i].copy())
            #xprint(f"The selected chromosome has a d2o of {sorted_population[i].d2o} and a fitness of {sorted_population[i].fitness:.2f}")
        # Part B: Probabilistic selection for the rest
        nombre_a_selectionner = tier_size - self.elitism
        #print (f"the number to select is { nombre_a_selectionner }")

        if nombre_a_selectionner > 0:
            pop_a_selectionee = sorted_population[self.elitism:]
            # Calculate probabilities proportional to fitness
            probas = self._calculer_probabilites_selection(pop_a_selectionee)
            #print(probas)
            #print(f"the first in the list has a d2o of {pop_a_selectionee[i].d2o} and a fitness of {pop_a_selectionee[i].fitness:.2f}")
            # Select with proportional probability
            for _ in range(nombre_a_selectionner):
                selected = random.choices(pop_a_selectionee, weights=probas, k=1)[0]
                #print( f"the selected ones have a d2o of {selected.d2o} and a fitness of {selected.fitness:.2f}")
                selectionnes.append(selected.copy())

        return selectionnes

    def _mutation_tier2(self,
                       selectionnes: List[Chromosome],
                       d2o_variation_rate: float) -> List[Chromosome]:
        """
        TIER 2: Generates n/3 chromosomes by mutation

        For each chromosome to generate:
        1. Choose a parent at random among the selected
        2. Mutate 1, 2, or 3 AA randomly
        3. 50% chance: mutate D2O (random variation)
        """
        tier_size = self.population_size // 3
        mutes = []

        while len(mutes) < tier_size:
            # 1. Choose a parent at random
            parent = random.choice(selectionnes)
            enfant = parent.copy()

            # 2. Mutate AA (1, 2, or 3 mutations)
            nombre_mutations = random.choice([1, 2, 3])

            # Get modifiable indices
            indices_modifiables = [i for i in range(len(self.aa_list)) if self.modifiable[i]]

            # Randomly select AA to mutate
            if len(indices_modifiables) >= nombre_mutations:
                indices_a_muter = random.sample(indices_modifiables, nombre_mutations)

                for idx in indices_a_muter:
                    enfant.deuteration[idx] = not enfant.deuteration[idx]

            # 3. Mutate D2O (50% chance)
            if random.choice([True, False]):
                enfant.modify_d2o(d2o_variation_rate)
            if self._unique_check(enfant, mutes + selectionnes):
                mutes.append(enfant)

        return mutes

    def _crossover_tier3(self, selectionnes: List[Chromosome]) -> List[Chromosome]:
        """
        TIER 3: Generates n/3 chromosomes by crossover

        For each chromosome to generate:
        1. Choose 2 parents at random
        2. Random cut point between 1 and 19
        3. Create child by crossover
        4. D2O transmission according to special rules
        """
        tier_size = self.population_size // 3
        crossovers = []

        while len(crossovers) < tier_size:
            # 1. Choose 2 parents at random
            parent1, parent2 = random.sample(selectionnes, 2)

            # 2. Random cut point (between 1 and 19)
            point_coupe = random.randint(1, len(self.aa_list) - 1)

            # 3. Create child by crossover
            enfant = Chromosome(self.aa_list, self.modifiable)

            # Crossover of deuteration vector
            enfant.deuteration = (
                parent1.deuteration[:point_coupe] +
                parent2.deuteration[point_coupe:]
            )

            # 4. D2O transmission according to rules
            if point_coupe == len(self.aa_list) // 2:  # Exactly at half (position 10 for 20 AA)
                # Transmit one of the 2 parents randomly
                enfant.d2o = random.choice([parent1.d2o, parent2.d2o])
            else:
                # Transmit with the parent of the smaller piece
                if point_coupe < len(self.aa_list) // 2:
                    enfant.d2o = parent1.d2o
                else:
                    enfant.d2o = parent2.d2o

            crossovers.append(enfant)

        return crossovers



# Usage example
if __name__ == "__main__":

    # Parse command-line arguments
    args = parse_arguments()

    # Set random seed if specified (for reproducibility)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Display configuration

    print("GENETIC ALGORITHM CONFIGURATION")
    print(f"Population size      : {args.population_size}")
    print(f"Elitism              : {args.elitism}")
    #print(f"Initial D2O          : {args.d2o_initial}%")
    print(f"D2O variation rate   : ±{args.d2o_variation_rate}%")
    print(f"Mutation rate        : {args.mutation_rate}")
    print(f"Crossover rate       : {args.crossover_rate}")
    print(f"Generations          : {args.generations}")


    # Create and execute the genetic algorithm
    print("\n>>> GENERATION 0 - Population creation")
    generator = PopulationGenerator(
        aa_list=AMINO_ACIDS,
        modifiable=restrictions,
        population_size=args.population_size,
        #d2o_initial=args.d2o_initial,
        elitism=args.elitism,
        d2o_variation_rate=args.d2o_variation_rate,
    )

    #Generation and display of initial population
    pop = generator.generate_initial_population()

    # Simulate random fitness scores (normally calculated by SANS)
    print("\n>>> Simulating SANS evaluation...")
    fitness_scores = [random.uniform(0.3, 0.9) for _ in pop]

    #assign fitness to chromosomes
    for chrom, fitness in zip(pop, fitness_scores):
        chrom.fitness = fitness

    sorted_pop = sorted(pop,
                               key=lambda x: x.fitness,
                               reverse=True)

    for i, chromosome in enumerate(sorted_pop):
        print(f"{i+1}. {chromosome}")



    # Generation n : Evolution
    for gen in range(1, args.generations ):
        print(f"\n>>> GENERATION {gen} - Population evolution")
        pop = generator.generate_next_generation(
            previous_population=sorted_pop,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            d2o_variation_rate=args.d2o_variation_rate
        )

        # Simulate random fitness scores (normally calculated by SANS)
        print("\n>>> Simulating SANS evaluation...")
        fitness_scores = [random.uniform(0.3, 0.9) for _ in pop]

        # assign fitness to chromosomes
        for chrom, fitness in zip(pop, fitness_scores):
            chrom.fitness = fitness

        sorted_pop = sorted(pop,
                              key=lambda x: x.fitness,
                              reverse=True)

        for i, chromosome in enumerate(sorted_pop):
            print(f"{i+1}. {chromosome}")
            if args.population_size / 3 == i+1:
                print("\n")
            if (args.population_size / 3 )*2== i+1:
                print("\n")