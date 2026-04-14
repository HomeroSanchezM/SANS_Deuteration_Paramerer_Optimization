"""
SANS DEUTERATION OPTIMIZATION

Population Generation Module
This module handles the creation and evolution of chromosome populations
representing amino acid deuteration configurations.

Architecture:
    - AminoAcid: Dataclass representing an amino acid
    - Chromosome: Class representing a solution (deuteration configuration)
    - PopulationGenerator: Main class managing generation/evolution

Usage:
    # First generation
    generator = PopulationGenerator(
        aa_list=EFFECTIVE_AMINO_ACIDS,
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
        d2o_variation_rate=5,
        new_generation=1
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
  python __init__.py --elitism 5
  python __init__.py -p 100 -e 3 -v 10
        """
    )

    # Optional config file (positional)
    parser.add_argument(
        "--config",
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
    parser.add_argument(
        '--d2o',
        type=int,
        nargs='+',  # accept one or more values
        default=None,
        help='If defined, blocks the d2o variation for a given list of d2o values'
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

    return parser.parse_args()


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

    if not (0 <= cfg["d2o_variation_rate"] <= 100):
        raise ValueError("d2o_variation_rate must be in [0, 100]")

    # Execution
    if cfg["generations"] < 1:
        raise ValueError("generations must be >= 1")

    # Restrictions
    # Accept both 18 (effective) and 20 (canonical) restriction lengths.
    # Auto-convert 20 -> 18 in place so downstream code always sees 18.
    restr_len = len(cfg["restrictions"])
    if restr_len == len(AMINO_ACIDS):                      # 20 -> convert
        cfg["restrictions"] = merge_restrictions_to_18(cfg["restrictions"])
    elif restr_len != N_EFFECTIVE_AA:
        raise ValueError(
            f"restrictions length ({restr_len}) must be "
            f"{N_EFFECTIVE_AA} (effective) or {len(AMINO_ACIDS)} (canonical)"
        )
    # D2O restrictions
    d2o = cfg.get("d2o")

    if d2o is not None:
        if not isinstance(d2o, list):
            raise TypeError("d2o must be a list of integers or None")

        if len(d2o) == 0:
            raise ValueError("d2o list cannot be empty")

        for value in d2o:
            if not isinstance(value, int):
                raise TypeError(f"d2o value {value} is not an integer")

            if not (0 <= value <= 100):
                raise ValueError(
                    f"d2o value {value} out of range [0, 100]"
                )


def load_config_ini(path: str):
    """
    Loads configuration from a .ini file.

    The [RESTRICTIONS] section is expected to have one boolean per canonical
    amino acid (20 entries with 3-letter codes as keys).  This function
    automatically converts the 20-entry vector to the 18-entry effective
    vector used by the chromosome, applying OR logic for linked pairs.
    """
    config = configparser.ConfigParser()
    config.read(path)

    # Parse D2O
    raw_d2o = config.get("D2O", "d2o", fallback="None").strip()

    if raw_d2o.lower() == "none":
        d2o_list = None
    else:
        try:
            d2o_list = [int(x) for x in raw_d2o.split()]
        except ValueError:
            raise ValueError(f"Invalid D2O list in config file: {raw_d2o}")

    # Read 20 per-AA restriction flags, then merge to 18 effective genes
    restrictions_20 = [
        config.getboolean("RESTRICTIONS", aa.code_3, fallback=True)
        for aa in AMINO_ACIDS
    ]
    restrictions_18 = merge_restrictions_to_18(restrictions_20)   # NEW

    cfg = {
        "population_size": config.getint("POPULATION", "population_size"),
        "elitism": config.getint("POPULATION", "elitism"),
        "d2o_variation_rate": config.getint("POPULATION",  "d2o_variation_rate"),
        "mutation_rate": config.getfloat("GENETIC", "mutation_rate"),
        "crossover_rate": config.getfloat("GENETIC", "crossover_rate"),
        "generations": config.getint("EXECUTION", "generations"),
        "seed": config.getint("EXECUTION", "seed", fallback=None),
        "restrictions":       restrictions_18,   # 18-element list (CHANGED from 20)
        "d2o":                d2o_list,
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
        "generations": pick(cli_args.generations, ini_cfg.get("generations"), 1),
        "seed": pick(cli_args.seed, ini_cfg.get("seed"), None),
        # Default: all 18 effective genes are modifiable
        "restrictions":       ini_cfg.get("restrictions", [True] * N_EFFECTIVE_AA),   # CHANGED
        "d2o":                pick(cli_args.d2o,               ini_cfg.get("d2o"),               None),
    }


# ============================================================================
#                   DATA STRUCTURE DEFINITIONS
# ============================================================================

@dataclass
class AminoAcid:
    """
    Represents an amino acid (or a linked pair of amino acids) with its codes.

    Attributes:
        name (str): Full name, e.g. "Alanine" or "Asparagine/Aspartic acid"
        code_3 (str): 3-letter code, e.g. "ALA" or "ASN+ASP" for linked pairs
        code_1 (str): 1-letter code, e.g. "A" or "ND" for linked pairs
    """
    name: str
    code_3: str
    code_1: str


# Full canonical list of 20 standard amino acids (used for PDB operations).
# indices are referenced throughout the codebase.
AMINO_ACIDS = [
    AminoAcid("Alanine",       "ALA", "A" ),  # index 0
    AminoAcid("Arginine",      "ARG", "R" ),  # index 1
    AminoAcid("Asparagine",    "ASN", "N" ),  # index 2
    AminoAcid("Aspartic acid", "ASP", "D" ),  # index 3
    AminoAcid("Cysteine",      "CYS", "C" ),  # index 4
    AminoAcid("Glutamic acid", "GLU", "E" ),  # index 5
    AminoAcid("Glutamine",     "GLN", "Q" ),  # index 6
    AminoAcid("Glycine",       "GLY", "G" ),  # index 7
    AminoAcid("Histidine",     "HIS", "H" ),  # index 8
    AminoAcid("Isoleucine",    "ILE", "I" ),  # index 9
    AminoAcid("Leucine",       "LEU", "L" ),  # index 10
    AminoAcid("Lysine",        "LYS", "K" ),  # index 11
    AminoAcid("Methionine",    "MET", "M" ),  # index 12
    AminoAcid("Phenylalanine", "PHE", "F" ),  # index 13
    AminoAcid("Proline",       "PRO", "P" ),  # index 14
    AminoAcid("Serine",        "SER", "S" ),  # index 15
    AminoAcid("Threonine",     "THR", "T" ),  # index 16
    AminoAcid("Tryptophan",    "TRP", "W" ),  # index 17
    AminoAcid("Tyrosine",      "TYR", "Y" ),  # index 18
    AminoAcid("Valine",        "VAL", "V" ),  # index 19
]

# Dictionary for quick access by 3-letter code
AA_DICT = {aa.code_3: i for i, aa in enumerate(AMINO_ACIDS)}

# LINKED AMINO ACID GROUPS
# Each entry is a list of 3-letter codes that always deuterate together.
# Linked pairs:
#   ASN + ASP  
#   GLU + GLN  
LINKED_AA_GROUPS: List[List[str]] = [
    ["ALA"],            # group  0  -> AMINO_ACIDS index 0
    ["ARG"],            # group  1  -> AMINO_ACIDS index 1
    ["ASN", "ASP"],     # group  2  -> AMINO_ACIDS indices 2, 3
    ["CYS"],            # group  3  -> AMINO_ACIDS index 4
    ["GLU", "GLN"],     # group  4  -> AMINO_ACIDS indices 5, 6
    ["GLY"],            # group  5  -> AMINO_ACIDS index 7
    ["HIS"],            # group  6  -> AMINO_ACIDS index 8
    ["ILE"],            # group  7  -> AMINO_ACIDS index 9
    ["LEU"],            # group  8  -> AMINO_ACIDS index 10
    ["LYS"],            # group  9  -> AMINO_ACIDS index 11
    ["MET"],            # group 10  -> AMINO_ACIDS index 12
    ["PHE"],            # group 11  -> AMINO_ACIDS index 13
    ["PRO"],            # group 12  -> AMINO_ACIDS index 14
    ["SER"],            # group 13  -> AMINO_ACIDS index 15
    ["THR"],            # group 14  -> AMINO_ACIDS index 16
    ["TRP"],            # group 15  -> AMINO_ACIDS index 17
    ["TYR"],            # group 16  -> AMINO_ACIDS index 18
    ["VAL"],            # group 17  -> AMINO_ACIDS index 19
]

# Dictionary 3-letter code -> group index in LINKED_AA_GROUPS
AA_GROUP_INDEX: dict = {}
for _gi, _grp in enumerate(LINKED_AA_GROUPS):
    for _code in _grp:
        AA_GROUP_INDEX[_code] = _gi


# EFFECTIVE_AMINO_ACIDS
# 18-element list, one entry per linked group.
# This is the list the Chromosome and PopulationGenerator work with.
EFFECTIVE_AMINO_ACIDS: List[AminoAcid] = []
for _grp in LINKED_AA_GROUPS:
    if len(_grp) == 1:
        # Standalone AA — use the canonical entry from AMINO_ACIDS
        EFFECTIVE_AMINO_ACIDS.append(
            next(aa for aa in AMINO_ACIDS if aa.code_3 == _grp[0])
        )
    else:
        # Linked pair — build a synthetic AminoAcid
        _aa_objects = [next(aa for aa in AMINO_ACIDS if aa.code_3 == c) for c in _grp]
        EFFECTIVE_AMINO_ACIDS.append(AminoAcid(
            name  = "/".join(aa.name   for aa in _aa_objects),   # e.g. "Asparagine/Aspartic acid"
            code_3= "+".join(aa.code_3 for aa in _aa_objects),   # e.g. "ASN+ASP"
            code_1= "".join(aa.code_1  for aa in _aa_objects),   # e.g. "ND"
        ))

# Number of effective genes in a chromosome (18)
N_EFFECTIVE_AA: int = len(EFFECTIVE_AMINO_ACIDS)   # 18



# expand 18-element vector -> 20-element vector 

def expand_deuteration_vector(deut_18: List[bool]) -> List[bool]:
    """
    Expand an 18-element chromosome deuteration vector to the full 20-element
    vector expected by the PDB deuteration engine.

    For linked groups (ASN+ASP, GLU+GLN), both amino acids in the pair receive
    the same deuteration state as their shared gene.

    Args:
        deut_18: 18-element boolean list (one entry per effective gene /
                 linked group in LINKED_AA_GROUPS).

    Returns:
        20-element boolean list aligned with AMINO_ACIDS ordering.

    Raises:
        ValueError: If the input length is neither 18 nor 20.

    Note:
        If a 20-element vector is passed it is returned unchanged

    """
    if len(deut_18) == 20:
        return list(deut_18)   # already full-length, pass through

    if len(deut_18) != N_EFFECTIVE_AA:
        raise ValueError(
            f"expand_deuteration_vector expects {N_EFFECTIVE_AA} or 20 elements, "
            f"got {len(deut_18)}"
        )

    deut_20 = [False] * len(AMINO_ACIDS)   # 20 elements, default False
    for gene_idx, group in enumerate(LINKED_AA_GROUPS):
        for aa_code in group:
            aa_idx = AA_DICT[aa_code]      # position in AMINO_ACIDS (0-19)
            deut_20[aa_idx] = deut_18[gene_idx]
    return deut_20


def merge_restrictions_to_18(restrictions_20: List[bool]) -> List[bool]:
    """
    Convert a 20-element restriction vector (one per canonical AA) to an
    18-element vector (one per effective gene / linked group).

    For linked pairs the group is considered deuterated if ANY member AA is
    deuterated in the input (OR logic).

    Args:
        restrictions_20: 20-element boolean list aligned with AMINO_ACIDS.

    Returns:
        18-element boolean list aligned with EFFECTIVE_AMINO_ACIDS.

    Note:
        If an 18-element list is passed it is returned unchanged.
    """
    if len(restrictions_20) == N_EFFECTIVE_AA:
        return list(restrictions_20)   # already 18 elements

    if len(restrictions_20) != len(AMINO_ACIDS):
        raise ValueError(
            f"merge_restrictions_to_18 expects {len(AMINO_ACIDS)} or "
            f"{N_EFFECTIVE_AA} elements, got {len(restrictions_20)}"
        )

    result: List[bool] = []
    for group in LINKED_AA_GROUPS:
        group_modifiable = any(restrictions_20[AA_DICT[code]] for code in group)
        result.append(group_modifiable)
    return result


# Default: all effective genes are modifiable
restrictions = [True] * N_EFFECTIVE_AA


# ============================================================================
#                           CHROMOSOME CLASS
# ============================================================================

class Chromosome:
    """
    Represents a chromosome (a deuteration solution).

    Linked pairs (ASN+ASP, GLU+GLN) always share the same deuteration state,
    so a single boolean controls both members of the pair.

    Attributes:
        aa_list (List[AminoAcid]): Effective AA list (18 entries, may include synthetic entries for linked pairs)
        modifiable (List[bool]):   18-element restriction list
        deuteration (List[bool]):  18-element deuteration vector (True = deuterated)
        d2o (int): Percentage of D2O (0-100)
        fitness (float): Fitness score (set by SANS evaluation)
        ratio (float): Ratio score (set by SANS evaluation)
        H (int): number of H atoms (set after pdb deuteration)
        D (int): number of D atoms (set after pdb deuteration)
        non_labile_D (int):        Non-labile D count (set after PDB deuteration)
        generation (int):          Generation in which this chromosome was first created.
                          Preserved when copied to a new generation (tier1).
        index (int): Index (1-based) of this chromosome in the population of its
                     creation generation. Preserved when copied to a new generation.
    """

    def __init__(self, aa_list: List[AminoAcid], modifiable: List[bool], fixed_d2o: List[int],
                 generation: int = 0, index: int = 0):
        """
        Initializes the chromosome with a random deuteration configuration and d2o.

        Args:
            aa_list:    Effective AA list
            modifiable: 18-element list indicating which effective genes are modifiable
            fixed_d2o:  Fixed list of allowed D2O values, or None for free variation
            generation: Generation number at creation (default 0)
            index:      1-based index within creation population (default 0)
        """
        self.aa_list = aa_list
        self.modifiable = modifiable
        self.generation = generation
        self.index = index
        self.randomize_deuteration()
        self.fixed_d2o = fixed_d2o
        self.d2o = random.choice(fixed_d2o) if fixed_d2o is not None else random.randint(0, 100)
        self.fitness = 0.0
        self.ratio  = 0
        self.H  = 0
        self.D  = 0
        self.non_labile_D = 0

    def copy(self) -> 'Chromosome':
        """
        Creates a deep copy of this chromosome.

        The copy retains the original ``generation`` and ``index`` values so that
        tier-1 (selected/elite) chromosomes keep their provenance across generations,
        which is also used to derive their stable PDB/SANS filenames.
        """
        new_chrom = Chromosome(self.aa_list, self.modifiable, self.fixed_d2o, self.generation, self.index)
        new_chrom.deuteration = self.deuteration[:]
        new_chrom.d2o = self.d2o
        new_chrom.fitness = self.fitness
        new_chrom.ratio = self.ratio
        new_chrom.H = self.H
        new_chrom.D = self.D
        new_chrom.non_labile_D = self.non_labile_D
        return new_chrom

    def randomize_deuteration(self) -> None:
        """
        Randomly initialises the 18-element deuteration vector.

        Rules:
        - Respects modifiable constraints
        - Each modifiable AA has 50% chance to be deuterated
        """
        n_genes = len(self.aa_list)          # 18 effective genes
        self.deuteration = [False] * n_genes

        # Choose how many genes to deuterate (0 to n_genes)
        nb_to_deuterate = random.randint(0, n_genes)

        # Collect modifiable indices and sample from them
        modifiable_indices = [i for i in range(n_genes) if self.modifiable[i]]
        if modifiable_indices:
            nb_to_deuterate = min(nb_to_deuterate, len(modifiable_indices))
            chosen = random.sample(modifiable_indices, nb_to_deuterate)
            for idx in chosen:
                self.deuteration[idx] = True

    def modify_d2o(self, variation_range: int = 10) -> None:
        """
        Modifies D2O value with random variation within limits.

        Args:
            variation_range: Maximum variation amplitude
        """
        max_down = min(variation_range, self.d2o)
        max_up   = min(variation_range, 100 - self.d2o)
        variation = random.randint(-max_down, max_up)
        self.d2o += variation

    def gaussian_modify_d2o(self, variation_range: int = 5) -> None:
        """
        Modifies D2O value with a binomial bell-curve distribution centred on 0.
	Without ends bias and more scrict range control

        Args:
            variation_range:  Maximum variation amplitude
        """
        sigma = variation_range / 2  # contain ~95% of values 

        while True:
            variation = int(round(random.gauss(0, sigma)))

            # range control
            if -variation_range <= variation <= variation_range:
                new_value = self.d2o + variation

                # limit end bias
                if 0 <= new_value <= 100:
                    self.d2o = new_value
                    return

    def __eq__(self, other: 'Chromosome') -> bool:
        """Checks identity based on deuteration vector + D2O (fitness excluded)."""
        if not isinstance(other, Chromosome):
            return False
        return self.deuteration == other.deuteration and self.d2o == other.d2o

    def __str__(self) -> str:
        """
        Human-readable representation.

        Linked pairs (e.g. ASN+ASP) appear as a single entry, making it clear
        that both AAs share the same deuteration state.
        """
        result = []
        for i, aa in enumerate(self.aa_list):
            tag = "(D)" if self.deuteration[i] else "(H)"
            result.append(f"{aa.code_3}{tag}")
        prov = f"[gen{self.generation:02d}_Chr{self.index:03d}]"
        return (prov + " " + " | ".join(result) +
                f" | D2O={self.d2o}% | Fitness={self.fitness:.4f}")

    def __repr__(self) -> str:
        return (f"Chromosome(gen={self.generation}, idx={self.index}, "
                f"d2o={self.d2o}, fitness={self.fitness:.4f})")

    def __hash__(self) -> int:
        """Hash based on deuteration vector + D2O (not fitness/generation/index)."""
        return hash((tuple(self.deuteration), self.d2o))


# ============================================================================
#                   POPULATION GENERATOR
# ============================================================================

class PopulationGenerator:
    """
    Main class managing the generation and evolution of chromosome populations.

    This class implements the complete genetic algorithm:
    - Initial generation 0: random population
    - Generation n: evolution via selection/mutation/crossover

    Attributes:
        aa_list (List[AminoAcid]): Effective AA list (18 entries)
        modifiable (List[bool]): 18-element restriction list
        population_size (int): Population size (MUST be multiple of 3)
        elitism (int):              Elite individuals preserved each generation
        d2o_variation_rate (int):   Maximum D2O mutation amplitude
        d2o (List[int] | None):     Fixed D2O values, or None for free variation
    """

    def __init__(self,
                 aa_list: List[AminoAcid],
                 modifiable: List[bool],
                 population_size: int,
                 elitism: int,
                 d2o_variation_rate: int,
                 d2o: List[int]):
        """
        Initializes the generator.

        Args:
            aa_list: List of amino acids
            modifiable: Modification constraints
            population_size: Total population size (MUST be multiple of 3)
            elitism: Number of elite individuals (≤ population_size/3)
            d2o_variation_rate: Maximum D2O variation amplitude

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
        self.d2o = d2o

    def generate_initial_population(self, generation: int = 0) -> List[Chromosome]:
        """
        Generates an initial population of unique chromosomes.

        Each chromosome is assigned:
        - ``generation`` = *generation* argument (default 0)
        - ``index``      = its 1-based position in the population list

        Args:
            generation: Generation number to stamp on each chromosome (default 0)

        Returns:
            List[Chromosome]: List of unique chromosomes
        """
        population = []
        attempts = 0
        max_attempts = self.population_size * 1000

        while len(population) < self.population_size and attempts < max_attempts:
            chrom = Chromosome(self.aa_list, self.modifiable,self.d2o)
            if self._unique_check(chrom, population):
                chrom.generation = generation
                chrom.index = len(population) + 1   # 1-based
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
                                 d2o_variation_rate: int = 5,
                                 new_generation: int = 1) -> List[Chromosome]:
        """
        Generates the next generation from the previous population.

        IMPORTANT: previous_population MUST have fitness values set.

        3-tier architecture:

            TIER 1 - SELECTION (n/3 chromosomes)
            - e best (elitism)
            - (n/3 - e) by probabilistic selection
            → Carries the original generation/index of its parent chromosome
              (stable filenames across generations).

            TIER 2 - MUTATION (n/3 chromosomes)
            - Random parent from tier 1
            - Mutate 1-5 effective genes
            - guassian probability: mutate D2O
            → Assigned generation=new_generation, index = tier_size+1 .. 2*tier_size

            TIER 3 - CROSSOVER (n/3 chromosomes)
            - 2 random parents from tier 1
            - Random cut point  on 18-gene vector (1-17)
            - D2O transmission according to special rules
            → Assigned generation=new_generation, index = 2*tier_size+1 .. 3*tier_size

        Args:
            previous_population: Must have fitness values set.
            d2o_variation_rate:   Maximum D2O variation amplitude
            new_generation:       Generation number to assign to tier-2/3 chromosomes

        Returns:
            List[Chromosome]: New population ordered as [tier1 | tier2 | tier3]
                              Fitness not yet set for tier-2/3.
        """
        if len(previous_population) != self.population_size:
            raise ValueError(f"Previous population size ({len(previous_population)}) "
                             f"!= expected size ({self.population_size})")

        # Sort by decreasing fitness for selection
        sorted_pop = sorted(previous_population, key=lambda x: x.fitness, reverse=True)

        tier_size = self.population_size // 3

        # TIER 1: Selection (n/3)  copies keep original generation/index
        tier1 = self._selection_tier1(sorted_pop)

        # TIER 2: Mutation (n/3)  new chromosomes
        tier2 = self._mutation_tier2(tier1, d2o_variation_rate, self.d2o)

        for i, chrom in enumerate(tier2):
            chrom.generation = new_generation
            chrom.index = tier_size + i + 1          # indices tier_size+1 .. 2*tier_size

        # TIER 3: Crossover (n/3)  new chromosomes
        tier3 = self._crossover_tier3(tier1, tier2, d2o_variation_rate, self.d2o )
        for i, chrom in enumerate(tier3):
            chrom.generation = new_generation
            chrom.index = 2 * tier_size + i + 1      # indices 2*tier_size+1 .. 3*tier_size

        new_population = tier1 + tier2 + tier3

        if len(new_population) != self.population_size:
            raise RuntimeError(
                f"New population has {len(new_population)} chromosomes, "
                f"expected {self.population_size}"
            )
        return new_population

    # ------------------------------------------------------------------
    #                       PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _unique_check(self, chromosome: Chromosome, population: List[Chromosome]) -> bool:
        """Returns True if chromosome is not already in population."""
        return chromosome not in population

    def _calcule_selection_probabilities(self, population: List[Chromosome]) -> List[float]:
        """Softmax-based selection probabilities to avoid 0% / 100% extremes."""
        fitness_array = np.array([c.fitness for c in population])
        fitness_array = fitness_array - fitness_array.min() + 1e-10
        temperature   = 0.5
        exp_fitness   = np.exp(fitness_array / temperature)
        probas        = exp_fitness / exp_fitness.sum()
        epsilon = 1e-6
        probas  = np.clip(probas, epsilon, 1.0 - epsilon)
        probas /= probas.sum()
        return probas.tolist()

    def _selection_tier1(self, sorted_population: List[Chromosome]) -> List[Chromosome]:
        """
        TIER 1: Selection of n/3 chromosomes.
        - e best (elitism) — copies with original generation/index
        - (n/3 - e) probabilistic — copies with original generation/index
        """
        tier_size = self.population_size // 3
        selects = []

        # Part A: Elitism
        for i in range(self.elitism):
            selects.append(sorted_population[i].copy())

        # Part B: Probabilistic selection
        if tier_size - self.elitism > 0:
            pop_to_select = sorted_population[self.elitism:]
            prob = self._calcule_selection_probabilities(pop_to_select)
            while len(selects) < tier_size:
                selected = random.choices(pop_to_select, weights=prob, k=1)[0]
                if selected not in selects:
                    selects.append(selected.copy())
        return selects

    def _mutation_tier2(self,
                        selectionnes: List[Chromosome],
                        d2o_variation_rate: float, d2o: List[int]) -> List[Chromosome]:
        """
        TIER 2: n/3 chromosomes produced by mutation.

        Each child inherits from a random tier-1 parent.
        Mutation flips 1-5 modifiable effective genes (18-gene space).
        D2O is shifted by a gaussian step (or sampled from the fixed list).
        """
        tier_size = self.population_size // 3
        mutes = []

        while len(mutes) < tier_size:
            parent = random.choice(selectionnes)
            enfant = parent.copy()

            # Flip 1-5 modifiable genes in the 18-gene space
            nombre_mutations    = random.choice([1, 2, 3, 4, 5])
            indices_modifiables = [i for i in range(len(self.aa_list))
                                   if self.modifiable[i]]
            if len(indices_modifiables) >= nombre_mutations:
                indices_a_muter = random.sample(indices_modifiables, nombre_mutations)
                for idx in indices_a_muter:
                    enfant.deuteration[idx] = not enfant.deuteration[idx]

            # D2O update
            if d2o is None:
                enfant.gaussian_modify_d2o(d2o_variation_rate)
            else :
                enfant.d2o = random.choice(d2o)

            if self._unique_check(enfant, mutes + selectionnes):
                mutes.append(enfant)

        return mutes

    def _crossover_tier3(self, selected: List[Chromosome],
                         mutes: List[Chromosome],
                         d2o_variation_rate: float,
                         d2o: List[int]) -> List[Chromosome]:
        """
        TIER 3: n/3 chromosomes produced by crossover.

        Two tier-1 parents are chosen; a random cut point in [1, 17] splits
        the 18-gene deuteration vectors.  D2O is randomised with a wider
        gaussian (2× variation_rate) or sampled from the fixed list.
        """
        tier_size  = self.population_size // 3
        crossovers = []

        while len(crossovers) < tier_size:
            parent1, parent2 = random.sample(selected, 2)
            # Cut point in [1, n_genes-1] — valid for 18-gene vector
            cut_point = random.randint(1, len(self.aa_list) - 1)

            enfant = Chromosome(self.aa_list, self.modifiable, self.d2o)
            enfant.deuteration = (
                parent1.deuteration[:cut_point] +
                parent2.deuteration[cut_point:]
            )

            if d2o is None:
                enfant.gaussian_modify_d2o(d2o_variation_rate * 2)
            else:
                enfant.d2o = random.choice(d2o)

            if self._unique_check(enfant, crossovers + mutes + selected):
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
    if cfg.get("d2o") is None :
        print(f"D2O variation rate   : ±{cfg['d2o_variation_rate']}%")
    else:
        d2o_msg = "D2O values           : " + " ".join(str(v) for v in cfg["d2o"])
        print(d2o_msg)
    print(f"Generations          : {cfg['generations']}")
    if cfg['seed'] is not None:
        print(f"Random seed          : {cfg['seed']}")
    print(f"Effective gene count : {N_EFFECTIVE_AA} (ASN+ASP and GLU+GLN linked)")
    print("=" * 50 + "\n")


def simulate_sans_evaluation(population: List[Chromosome]) -> List[Chromosome]:
    """
    Simulates SANS evaluation by assigning random fitness scores.
    """
    print(">>> Simulating SANS evaluation...")
    for chrom in population:
        chrom.fitness = random.uniform(0.3, 0.9)
    return population


def sort_and_display_population(population: List[Chromosome],
                                population_size: int,
                                generation: int = 1) -> List[Chromosome]:
    """
    Sorts population by fitness and displays it with tier separators.
    Returns sorted population (best first).
    """
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

    print(f"\n>>> GENERATION {generation} - Results")
    tier_size = population_size // 3

    for i, chromosome in enumerate(sorted_pop, 1):
        print(f"{i:2d}. {chromosome}")
        if i == tier_size or i == tier_size * 2:
            print("-" * 80)

    return sorted_pop


def run_genetic_algorithm(cfg: dict):
    """Main execution function for the genetic algorithm (standalone/demo mode)."""
    generator = PopulationGenerator(
        aa_list=EFFECTIVE_AMINO_ACIDS,         # 18 effective genes
        modifiable=cfg["restrictions"],         # 18-element list
        population_size=cfg["population_size"],
        elitism=cfg["elitism"],
        d2o_variation_rate=cfg["d2o_variation_rate"],
        d2o=cfg["d2o"]
    )

    print("\n>>> GENERATION 1 - Initial population creation")
    population = generator.generate_initial_population(generation=0)
    population = simulate_sans_evaluation(population)
    sorted_pop = sort_and_display_population(population, cfg["population_size"], generation=1)

    for gen in range(2, cfg["generations"] + 1):
        print(f"\n{'='*80}")
        print(f">>> GENERATION {gen} - Population evolution")
        print('='*80)

        population = generator.generate_next_generation(
            previous_population=sorted_pop,
            mutation_rate=cfg["mutation_rate"],
            crossover_rate=cfg["crossover_rate"],
            d2o_variation_rate=cfg["d2o_variation_rate"],
            new_generation=gen - 1
        )

        population = simulate_sans_evaluation(population)
        sorted_pop = sort_and_display_population(population, cfg["population_size"], generation=gen)

    print("\n" + "="*80)
    print("ALGORITHM COMPLETED")
    print("="*80)
    print(f"Best solution: {sorted_pop[0]}")
    print(f"Expanded 20-AA vector: {expand_deuteration_vector(sorted_pop[0].deuteration)}")
    print("=" * 80 + "\n")


# ============================================================================
#                           MAIN EXECUTION
# ============================================================================

def main():
    args   = parse_arguments()
    ini_cfg = load_config_ini(args.config) if args.config else None
    cfg    = merge_config(args, ini_cfg)

    try:
        validate_config(cfg)
    except ValueError as e:
        print(f"\n CONFIGURATION ERROR")
        print(f"   {e}\n")
        exit(1)

    if cfg["seed"] is not None:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

    display_config(cfg)
    run_genetic_algorithm(cfg)


if __name__ == "__main__":
    main()
