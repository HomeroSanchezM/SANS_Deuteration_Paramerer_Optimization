#!/usr/bin/env python3
"""
Main script to generate a chromosome population and create deuterated PDBs
==================================================================================

This script combines chromosome generation and PDB structure deuteration.

Usage:
    python generate_deuterated_pdbs.py <fichier.pdb> -g 1 -p 30
    python generate_deuterated_pdbs.py input.pdb --population_size 60 --elitism 5

Arguments:
    file.pdb          : Source PDB file to deuterate
    -g, --generations    : Number of generations (default: 1)
    -p, --population_size: Population size (multiple of 3, default: 6)
    -e, --elitism        : Number of elites (default: 2)
    -v, --d2o_variation_rate: Maximum variation of D2O (default: 5)
    -v, --d2o_variation_rate: Maximum D2O variation (default: 5)
    -m, --mutation_rate  : Mutation rate (default: 0.15)
    -c, --crossover_rate : Crossover rate (default: 0.8)
    --seed               : Random seed for reproducibility
    --output_dir         : Output folder (default: deuterated_pdb)

output:
    - The folder deuterated_pdb/ contains all the generated PDBs.
    - population_summary.txt file with details of each chromosome
"""

import sys
import argparse
import random
import logging
from pathlib import Path
from typing import List
import numpy as np

from __init__ import (
    AMINO_ACIDS,
    restrictions,
    PopulationGenerator,
    Chromosome
)
from pdb_deuteration import PdbDeuteration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Chromosome population generation and creation of deuterated PDBs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python generate_deuterated_pdbs.py structure.pdb -g 1 -p 30
  python generate_deuterated_pdbs.py input.pdb --population_size 60 --elitism 5 --seed 42
  python generate_deuterated_pdbs.py myprotein.pdb -p 90 -e 3 -v 10 --output_dir results
        """
    )

    # Required argument
    parser.add_argument(
        'pdb_file',
        type=str,
        help='Source PDB file to deuterate'
    )

    # Population parameters
    parser.add_argument(
        '-p', '--population_size',
        type=int,
        default=6,
        help='Population size (multiple of 3, default: 6)'
    )

    parser.add_argument(
        '-e', '--elitism',
        type=int,
        default=2,
        help='Number of elites (default: 2)'
    )

    # D2O parameters
    parser.add_argument(
        '-v', '--d2o_variation_rate',
        type=int,
        default=5,
        help='Maximum variation of D2O (default: 5)'
    )

    # Genetic parameters
    parser.add_argument(
        '-m', '--mutation_rate',
        type=float,
        default=0.15,
        help='Mutation rate (0.0-1.0). Default: 0.15'
    )

    parser.add_argument(
        '-c', '--crossover_rate',
        type=float,
        default=0.8,
        help='Crossover rate (0.0-1.0). Default: 0.8'
    )

    # Runtime parameters
    parser.add_argument(
        '-g', '--generations',
        type=int,
        default=1,
        help=' Number of generations (default: 1)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed for reproducibility (defaut: 1)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output folder  (defaut: strcuture_name_deuterated_pdb/)'
    )

    args = parser.parse_args()

    # Argument Validation
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


def create_output_directory(output_dir: str, pdb_file: str) -> Path:
    """
    Creates the output folder if it doesn't already exist.

    Args:
        output_dir: Path to the output folder
    Returns:
        Path: Path object of the created folder

    """
    if output_dir is None:
        output_path = Path(pdb_file.split(".")[0]+"_deuterated_pdbs")
        ref_path = output_path / "ref"
        output_path.mkdir(exist_ok=True, parents=True)
        ref_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"Output folder created/verified: {output_path.absolute()}")

    else:
        output_path = Path(output_dir)
        ref_path = output_path / "ref"
        output_path.mkdir(exist_ok=True, parents=True)
        ref_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"Output folder created/verified: {output_path.absolute()}")
    return output_path, ref_path


def generate_pdb_filename(chromosome: Chromosome, index: int, generation: int = 0) -> str:
    """
    Generates a descriptive filename for the PDB.

    Format: gen{G}_chr{N}_d2o{XX}_deutAA{Y}.pdb
    where:
        G = generation number
        N = chromosome number
        XX = percentage of D2O
        Y = number of deuterated AA

    Args:
        chromosome: The chromosome to name
        index: Index of the chromosome in the population
        generation: Generation number

    Returns:
        str: Filename

    """
    num_deuterated_aa = sum(chromosome.deuteration)
    return f"gen{generation+1:02d}_chr{index:03d}_d2o{chromosome.d2o:02d}_deutAA{num_deuterated_aa:02d}.pdb"


def save_population_summary(population: List[Chromosome],
                            output_dir: Path,
                            generation: int = 0) -> None:
    """
    Saves a summary of the population to a text file.

    Args:
        population: List of chromosomes
        output_dir: Output directory
        generation: Generation number

    """
    summary_file = output_dir / f"generation_{generation:02d}_summary.txt"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"POPULATION SUMMARY - GENERATION {generation}\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"Population size : {len(population)}\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'#':<5} {'PDB Filename':<45} {'D2O%':<8} {'AA deut.':<12} {'Fitness':<10}\n")
        f.write("-" * 80 + "\n")

        for i, chromosome in enumerate(population, 1):
            filename = generate_pdb_filename(chromosome, i, generation)
            num_deut = sum(chromosome.deuteration)
            fitness_str = f"{chromosome.fitness:.4f}" if chromosome.fitness is not None else "N/A"
            f.write(f"{i:<5} {filename:<45} {chromosome.d2o:<8} {num_deut:<12} {fitness_str:<10}\n")

        f.write("-" * 80 + "\n\n")

        # Statistics
        d2o_values = [c.d2o for c in population]
        deut_counts = [sum(c.deuteration) for c in population]

        f.write("STATISTICS:\n")
        f.write(f"  Average D2O: {np.mean(d2o_values):.2f}% (min: {min(d2o_values)}%, max: {max(d2o_values)}%)\n")
        f.write(f"  Average deuterated AA: {np.mean(deut_counts):.2f} (min: {min(deut_counts)}, max: {max(deut_counts)})\n")

        if any(c.fitness is not None for c in population):
                fitness_values = [c.fitness for c in population if c.fitness is not None]
                f.write(f"  Fitness moyen: {np.mean(fitness_values):.4f} "
                        f"(min: {min(fitness_values):.4f}, max: {max(fitness_values):.4f})\n")

        f.write("\n" + "=" * 80 + "\n")

    logger.info(f"Population summary saved: {summary_file}")


def simulate_fitness_evaluation(population: List[Chromosome]) -> None:
    """
    Simulate fitness evaluation for the population.

    In production, this would be replaced by actual SANS calculations.
    For now, we assign random fitness values.

    Args:
        population: List of chromosomes to evaluate
    """
    logger.info(">>> Simulating SANS evaluation (assigning random fitness)...")

    for chromosome in population:
        # Simulate fitness score between 0.3 and 0.9
        chromosome.fitness = random.uniform(0.3, 0.9)

def generate_deuterated_pdbs(pdb_file: str,
                             population: List[Chromosome],
                             output_dir: Path,
                             generation: int = 0) -> None:
    """
    Generate all deuterated PDB files for a population.

    Args:
        pdb_file: Path to source PDB file
        population: List of chromosomes
        output_dir: Output directory
        generation: Generation number
    """
    logger.info(f"Generating {len(population)} deuterated PDB files...")

    for i, chromosome in enumerate(population, 1):
        try:
            # Load a new PDB instance for each chromosome
            deuterator = PdbDeuteration(pdb_file)

            # Apply deuteration
            deuterator.apply_deuteration(chromosome)

            # Generate filename
            output_filename = generate_pdb_filename(chromosome, i, generation)
            output_path = output_dir / output_filename

            # Save deuterated PDB
            deuterator.save(str(output_path))

            logger.info(f"  [{i}/{len(population)}] Generated: {output_filename}")

        except Exception as e:
            logger.error(f"  [{i}/{len(population)}] ERROR generating {output_filename}: {e}")
            continue

    logger.info(f"Generation complete! {len(population)} PDB files created in {output_dir}")


def display_population_summary(population: List[Chromosome], generation: int) -> None:
    """
    Display a brief summary of the population in the console.

    Args:
        population: List of chromosomes
        generation: Generation number
    """
    logger.info(f"\nGeneration {generation} - Top 3 chromosomes:")

    # Sort by fitness (descending)
    sorted_pop = sorted(population, key=lambda x: x.fitness if x.fitness else 0, reverse=True)

    for i in range(min(3, len(sorted_pop))):
        chrom = sorted_pop[i]
        num_deut = sum(chrom.deuteration)
        fitness_str = f"{chrom.fitness:.4f}" if chrom.fitness else "N/A"
        logger.info(f"  {i + 1}. D2O={chrom.d2o:2d}%, AA deut={num_deut:2d}/20, Fitness={fitness_str}")

    if len(sorted_pop) > 3:
        logger.info(f"  ... and {len(sorted_pop) - 3} other chromosomes")


def main():
    """Main function of the script."""

    # Parse arguments
    args = parse_arguments()

    # Check if PDB file exists
    pdb_path = Path(args.pdb_file)
    if not pdb_path.exists():
        logger.error(f"PDB file not found: {args.pdb_file}")
        sys.exit(1)

    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")

    # Display configuration
    logger.info("=" * 80)
    logger.info("DEUTERATED PDB GENERATION")
    logger.info("=" * 80)
    logger.info(f"Source PDB file      : {args.pdb_file}")
    logger.info(f"Population size      : {args.population_size}")
    logger.info(f"Elitism              : {args.elitism}")
    logger.info(f"D2O variation rate   : ±{args.d2o_variation_rate}%")
    logger.info(f"Mutation rate        : {args.mutation_rate}")
    logger.info(f"Crossover rate       : {args.crossover_rate}")
    logger.info(f"Generations          : {args.generations}")
    logger.info(f"Output directory     : {args.output_dir}")
    logger.info("=" * 80)

    # Create output directory
    output_dir, ref_dir = create_output_directory(args.output_dir, pdb_path.name)

    # Create ref pdb
    logger.info("\n>>> REF - Creating ref pbd")
    total_deuteration = Chromosome(
        aa_list=AMINO_ACIDS,
        modifiable=restrictions
    )
    total_deuteration.deuteration = [True]*20
    total_deuteration.d2o = 100

    total_protonation = Chromosome(
        aa_list=AMINO_ACIDS,
        modifiable=restrictions
    )
    total_protonation.protonation = [False]*20
    total_protonation.d2o = 0

    # Generate red PDBs
    logger.info("Generating total deuteration PDB files")
    total_deuteration_deuterator = PdbDeuteration(args.pdb_file)
    total_deuteration_deuterator.apply_deuteration(total_deuteration)
    total_deuteration_deuterator.save(ref_dir / Path(pdb_path.name.split(".")[0] + "_total_deuteration.pdb") )

    logger.info("Generating total protonation PDB files")
    total_protonation_deuterator = PdbDeuteration(args.pdb_file)
    total_protonation_deuterator.apply_deuteration(total_protonation)
    total_protonation_deuterator.save(ref_dir / Path(pdb_path.name.split(".")[0] + "_total_protonation.pdb"))

    # Create population generator
    logger.info("\n>>> GENERATION 0 - Creating initial population")
    generator = PopulationGenerator(
        aa_list=AMINO_ACIDS,
        modifiable=restrictions,
        population_size=args.population_size,
        elitism=args.elitism,
        d2o_variation_rate=args.d2o_variation_rate,
    )

    # Generate initial population
    population = generator.generate_initial_population()
    logger.info(f"Initial population generated: {len(population)} chromosomes")

    # Simulate fitness evaluation for generation 0
    simulate_fitness_evaluation(population)

    # Sort population by fitness
    sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)

    # Display summary
    display_population_summary(sorted_population, generation=0)

    # Save population summary
    save_population_summary(sorted_population, output_dir, generation=0)

    # Generate deuterated PDBs
    logger.info("\n>>> Generating deuterated PDB files for generation 0")
    generate_deuterated_pdbs(
        pdb_file=args.pdb_file,
        population=sorted_population,
        output_dir=output_dir,
        generation=0
    )

    # Process additional generations
    for gen in range(1, args.generations):
        logger.info("\n" + "=" * 80)
        logger.info(f">>> GENERATION {gen} - Population evolution")
        logger.info("=" * 80)

        # Generate next generation
        population = generator.generate_next_generation(
            previous_population=sorted_population,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            d2o_variation_rate=args.d2o_variation_rate
        )

        logger.info(f"New population generated: {len(population)} chromosomes")

        # Simulate fitness evaluation
        simulate_fitness_evaluatgition(population)

        # Sort population by fitness
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)

        # Display summary
        display_population_summary(sorted_population, generation=gen)

        # Save population summary
        save_population_summary(sorted_population, output_dir, generation=gen)

        # Generate deuterated PDBs
        logger.info(f"\n>>> Generating deuterated PDB files for generation {gen}")
        generate_deuterated_pdbs(
            pdb_file=args.pdb_file,
            population=sorted_population,
            output_dir=output_dir,
            generation=gen
        )

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PROCESS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"Total generations processed: {args.generations}")
    logger.info(f"Total PDB files generated: {args.generations * args.population_size}")
    logger.info(f"Files generated in: {output_dir.absolute()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        sys.exit(1)