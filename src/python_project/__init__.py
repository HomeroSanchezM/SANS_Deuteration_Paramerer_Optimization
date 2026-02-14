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
    population = generator.generate_initial_population()

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
import configparser
from typing import List, Tuple
from dataclasses import dataclass


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
  python __init__.py -p 100 -e 3 -m 0.15 -c 0.8 -v 10
        """
    )

    # Optional config file (positional)
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to config.ini file"
    )

    # Population parameters
    parser.add_argument(
        '-p', '--population_size',
        type=int,
        help='Population size (number of chromosomes, MUST be a multiple of 3). Default: 30'
    )

    parser.add_argument(
        '-e', '--elitism',
        type=int,
        help='Number of elite individuals preserved at each generation (must be ≤ population_size/3). Default: 5'
    )

    parser.add_argument(
        '-v', '--d2o_variation_rate',
        type=int,
        help='Maximum D2O variation amplitude. Default: 50'
    )

    # Genetic parameters
    parser.add_argument(
        '-m', '--mutation_rate',
        type=float,
        help='Mutation rate (0.0-1.0). Probability that a gene will be mutated. Default: 0.15'
    )

    parser.add_argument(
        '-c', '--crossover_rate',
        type=float,
        help='Crossover rate (0.0-1.0). Probability of crossover between two parents. Default: 0.8'
    )

    # Execution parameters
    parser.add_argument(
        '-g', '--generations',
        type=int,
        help='Number of generations to execute. Default: 1'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility. Default: None (random)'
    )

    args = parser.parse_args()

    return args

def validate_config(cfg):
    """Validates the configuration parameters."""
    # Population
    if cfg["population_size"] <= 0:
        raise ValueError("population_size must be > 0")

    if cfg["population_size"] % 3 != 0:
        raise ValueError(
            f"population_size must be a multiple of 3 "
            f"(got {cfg['population_size']})"
        )

    if cfg["elitism"] < 0:
        raise ValueError("elitism must be >= 0")

    if cfg["elitism"] > cfg["population_size"] // 3:
        raise ValueError(
            f"elitism must be ≤ population_size/3 "
            f"(max {cfg['population_size'] // 3})"
        )

    # Rates
    if not (0.0 <= cfg["mutation_rate"] <= 1.0):
        raise ValueError("mutation_rate must be in [0.0, 1.0]")

    if not (0.0 <= cfg["crossover_rate"] <= 1.0):
        raise ValueError("crossover_rate must be in [0.0, 1.0]")

    if not (0 <= cfg["d2o_variation_rate"] <= 100):
        raise ValueError("d2o_variation_rate must be in [0, 100]")

    # Execution
    if cfg["generations"] < 1:
        raise ValueError("generations must be >= 1")

    # Restrictions
    if len(cfg["restrictions"]) != len(AMINO_ACIDS):
        raise ValueError(
            f"restrictions length ({len(cfg['restrictions'])}) "
            f"!= number of amino acids ({len(AMINO_ACIDS)})"
        )


def load_config_ini(path: str):
    """Loads configuration from a .ini file."""
    config = configparser.ConfigParser()
    config.read(path)

    cfg = {
        "population_size": config.getint("POPULATION", "population_size"),
        "elitism": config.getint("POPULATION", "elitism"),
        "d2o_variation_rate": config.getint("POPULATION", "d2o_variation_rate"),
        "mutation_rate": config.getfloat("GENETIC", "mutation_rate"),
        "crossover_rate": config.getfloat("GENETIC", "crossover_rate"),
        "generations": config.getint("EXECUTION", "generations"),
        "seed": config.getint("EXECUTION", "seed", fallback=None),
        "restrictions": [
            config.getboolean("RESTRICTIONS", aa.code_3)
            for aa in AMINO_ACIDS
        ]
    }
    return cfg

def merge_config(cli_args, ini_cfg=None):
    """Merges CLI arguments with INI configuration (CLI takes precedence)."""
    ini_cfg = ini_cfg or {}

    def pick(cli, ini, default):
        return cli if cli is not None else ini if ini is not None else default

    return {
        "population_size": pick(cli_args.population_size, ini_cfg.get("population_size"), 30),
        "elitism": pick(cli_args.elitism, ini_cfg.get("elitism"), 2),
        "d2o_variation_rate": pick(cli_args.d2o_variation_rate, ini_cfg.get("d2o_variation_rate"), 5),
        "mutation_rate": pick(cli_args.mutation_rate, ini_cfg.get("mutation_rate"), 0.15),
        "crossover_rate": pick(cli_args.crossover_rate, ini_cfg.get("crossover_rate"), 0.8),
        "generations": pick(cli_args.generations, ini_cfg.get("generations"), 1),
        "seed": pick(cli_args.seed, ini_cfg.get("seed"), None),
        "restrictions": ini_cfg.get("restrictions", [True] * len(AMINO_ACIDS))
    }


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

restrictions = [True] * len(AMINO_ACIDS)
# ============================================================================
#                           CHROMOSOME CLASS
# ============================================================================

class Chromosome:
    """
    Represents a chromosome (a deuteration solution).

    Attributes:
        aa_list (List[AminoAcid]): List of amino acids
        modifiable (List[bool]): Restriction list (which AA are modifiable)
        deuteration (List[bool]): Deuteration vector (True = deuterated)
        d2o (int): Percentage of D2O (0-100)
        fitness (float): Fitness score (set by SANS evaluation)
    """

    def __init__(self, aa_list: List[AminoAcid], modifiable: List[bool]):
        """
        Initializes the chromosome with a random deuteration configuration and d20 configuration.

        Args:
            aa_list: List of amino acids
            modifiable: List indicating which AA are modifiable
        """
        self.aa_list = aa_list
        self.modifiable = modifiable
        self.randomize_deuteration()
        self.d2o = random.randint(0, 100)# Default value
        self.fitness = 0.0  # Will be set after SANS evaluation

    def copy(self) -> 'Chromosome':
        """Creates a deep copy of this chromosome."""
        new_chrom = Chromosome(self.aa_list, self.modifiable)
        new_chrom.deuteration = self.deuteration[:]
        new_chrom.d2o = self.d2o
        new_chrom.fitness = self.fitness
        return new_chrom

    def randomize_deuteration(self) -> None:
        """
        Randomly initializes the deuteration vector.

        Rules:
        - Respects modifiable constraints
        - Each modifiable AA has 50% chance to be deuterated
        """
        self.deuteration = [False] * len(self.aa_list) # All non-deuterated by default
        for i in range(len(self.aa_list)):
            if self.modifiable[i]:
                self.deuteration[i] = random.choice([True, False])

    def modify_d2o(self, variation_range: int = 10) -> None:
        """
        Modifies D2O value with random variation within limits.

        Args:
            variation_range: Maximum variation amplitude

        Rules:
        - New value: current ± [0, variation_range]
        - Constrained to [0, 100]
        """
        variation = random.randint(-variation_range, variation_range)
        self.d2o = max(0, min(100, self.d2o + variation))

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

    def __str__(self) -> str:
        """
        Readable representation showing deuterated amino acids.

        Returns:
            str: "{AA codes} | D2O: X% | Fitness: Y"
        """
        result = []
        for i, aa in enumerate(self.aa_list):
            if self.deuteration[i]:
                result.append(f"{aa.code_3}(D)")
            else:
                result.append(f"{aa.code_3}(H)")
        return " | ".join(result) + f" | D2O={self.d2o}% | Fitness={self.fitness:.2f}"

    def __repr__(self) -> str:
        """Technical representation."""
        return f"Chromosome(d2o={self.d2o}, fitness={self.fitness:.2f})"


    def __hash__(self) -> int:
        """
        Allows chromosomes to be used in sets/dicts.

        Hash based on deuteration + D2O (not fitness)
        """
        return hash((tuple(self.deuteration), self.d2o))


# ============================================================================
#                   POPULATION GENERATOR
# ============================================================================

class PopulationGenerator:
    """
    Main class managing the generation and evolution of populations.

    This class implements the complete genetic algorithm:
    - Initial generation 0: random population
    - Generation n: evolution via selection/mutation/crossover

    Attributes:
        aa_list (List[AminoAcid]): List of amino acids
        modifiable (List[bool]): Restriction list
        population_size (int): Population size (MUST be multiple of 3)
        elitism (int): Number of elites preserved
        d2o_variation_rate (float): D2O variation rate for mutations
    """

    def __init__(self,
                 aa_list: List[AminoAcid],
                 modifiable: List[bool],
                 population_size: int,
                 elitism: int,
                 d2o_variation_rate: int):
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
        self.elitism = elitism
        self.d2o_variation_rate = d2o_variation_rate

    def generate_initial_population(self) -> List[Chromosome]:
        """
        Generates an initial population of unique chromosomes.

        Returns:
            List[Chromosome]: List of unique chromosomes (not yet evaluated)
        """
        population = []
        attempts = 0
        max_attempts = self.population_size * 1000

        while len(population) < self.population_size and attempts < max_attempts:
            chrom = Chromosome(self.aa_list, self.modifiable)
            if self._unique_check(chrom, population):
                population.append(chrom)
            attempts += 1

        if len(population) < self.population_size:
            raise RuntimeError(
                f"Unable to generate {self.population_size} unique chromosomes "
                f"after {max_attempts} attempts. Only {len(population)} generated."
            )

        return population

    def generate_next_generation(self,
                                previous_population: List[Chromosome],
                                mutation_rate: float = 0.15,
                                crossover_rate: float = 0.8,
                                d2o_variation_rate: int = 5) -> List[Chromosome]:
        """
        Generates the next generation from the previous population.

        IMPORTANT: previous_population MUST have fitness values set

        3-tier architecture:

            TIER 1 - SELECTION (n/3 chromosomes)
            - e best (elitism)
            - (n/3 - e) by probabilistic selection
            → Serves as parents for tiers 2 and 3

            TIER 2 - MUTATION (n/3 chromosomes)
            - Random parent from tier 1
            - Mutate 1, 2, or 3 AA
            - 50% chance: mutate D2O

            TIER 3 - CROSSOVER (n/3 chromosomes)
            - 2 random parents from tier 1
            - Random cut point (1-19)
            - D2O transmission according to special rules


        Args:
            previous_population: Previous generation (sorted by fitness)
            mutation_rate: Mutation probability [0.0, 1.0]
            crossover_rate: Crossover probability [0.0, 1.0]
            d2o_variation_rate: Maximum D2O variation amplitude

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
	# TIER 1: Selection (n/3)
        tier1 = self._selection_tier1(sorted_pop)
	# TIER 2: Mutation (n/3)
        tier2 = self._mutation_tier2(tier1, d2o_variation_rate)
	# TIER 3: Crossover (n/3)
        tier3 = self._crossover_tier3(tier1)

        # Combine all tiers
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
        return chromosome not in population

    def _calcule_selection_probabilities(self, population: List[Chromosome]) -> List[float]:
        """
        Calculates selection probabilities proportional to fitness.

        IMPORTANT: Probabilities are NEVER 0% nor 100%

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
            probas = self._calcule_selection_probabilities(pop_a_selectionee)
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


# ============================================================================
#                   UTILITY FUNCTIONS
# ============================================================================

def display_config(cfg: dict):
    """Displays the genetic algorithm configuration."""
    print("\n" + "="*50)
    print("GENETIC ALGORITHM CONFIGURATION")
    print("="*50)
    print(f"Population size      : {cfg['population_size']}")
    print(f"Elitism              : {cfg['elitism']}")
    print(f"D2O variation rate   : ±{cfg['d2o_variation_rate']}%")
    print(f"Mutation rate        : {cfg['mutation_rate']}")
    print(f"Crossover rate       : {cfg['crossover_rate']}")
    print(f"Generations          : {cfg['generations']}")
    if cfg['seed'] is not None:
        print(f"Random seed          : {cfg['seed']}")
    print("="*50 + "\n")


def simulate_sans_evaluation(population: List[Chromosome]) -> List[Chromosome]:
    """
    Simulates SANS evaluation by assigning random fitness scores.
    
    Args:
        population: List of chromosomes to evaluate
        
    Returns:
        Population with assigned fitness scores
    """
    print(">>> Simulating SANS evaluation...")
    fitness_scores = [random.uniform(0.3, 0.9) for _ in population]
    
    for chrom, fitness in zip(population, fitness_scores):
        chrom.fitness = fitness
    
    return population


def sort_and_display_population(population: List[Chromosome], 
                                population_size: int,
                                generation: int = 1) -> List[Chromosome]:
    """
    Sorts population by fitness and displays it with tier separators.
    
    Args:
        population: List of chromosomes
        population_size: Size of the population (for tier calculation)
        generation: Generation number for display
        
    Returns:
        Sorted population (best first)
    """
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
    
    print(f"\n>>> GENERATION {generation} - Results")
    tier_size = population_size // 3
    
    for i, chromosome in enumerate(sorted_pop, 1):
        print(f"{i:2d}. {chromosome}")
        
        # Add separator between tiers
        if i == tier_size or i == tier_size * 2:
            print("-" * 80)
    
    return sorted_pop


def run_genetic_algorithm(cfg: dict):
    """
    Main execution function for the genetic algorithm.
    
    Args:
        cfg: Configuration dictionary
    """
    # Create the population generator
    generator = PopulationGenerator(
        aa_list=AMINO_ACIDS,
        modifiable=cfg["restrictions"],
        population_size=cfg["population_size"],
        elitism=cfg["elitism"],
        d2o_variation_rate=cfg["d2o_variation_rate"]
    )
    
    # Generate initial population (Generation 1)
    print("\n>>> GENERATION 1 - Initial population creation")
    population = generator.generate_initial_population()
    
    # Evaluate and display
    population = simulate_sans_evaluation(population)
    sorted_pop = sort_and_display_population(
        population, 
        cfg["population_size"], 
        generation=1
    )
    
    # Evolution through generations
    for gen in range(2, cfg["generations"]+1 ):
        print(f"\n{'='*80}")
        print(f">>> GENERATION {gen} - Population evolution")
        print('='*80)
        
        # Generate next generation
        population = generator.generate_next_generation(
            previous_population=sorted_pop,
            mutation_rate=cfg["mutation_rate"],
            crossover_rate=cfg["crossover_rate"],
            d2o_variation_rate=cfg["d2o_variation_rate"]
        )
        
        # Evaluate and display
        population = simulate_sans_evaluation(population)
        sorted_pop = sort_and_display_population(
            population,
            cfg["population_size"],
            generation=gen
        )
    
    # Final summary
    print("\n" + "="*80)
    print("ALGORITHM COMPLETED")
    print("="*80)
    print(f"Best solution: {sorted_pop[0]}")
    print("="*80 + "\n")


# ============================================================================
#                           MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point of the program."""
    # Parse arguments and load configuration
    args = parse_arguments()
    
    ini_cfg = None
    if args.config:
        ini_cfg = load_config_ini(args.config)
    
    cfg = merge_config(args, ini_cfg)
    
    # Validate configuration
    try:
        validate_config(cfg)
    except ValueError as e:
        print(f"\n CONFIGURATION ERROR")
        print(f"   {e}\n")
        exit(1)
    
    # Set random seed if specified
    if cfg["seed"] is not None:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
    
    # Display configuration
    display_config(cfg)
    
    # Run the genetic algorithm
    run_genetic_algorithm(cfg)


if __name__ == "__main__":
    main()
