import numpy as np
import argparse

def parse_arguments():
    """
    Parses command-line arguments to configure the genetic algorithm.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="fitness evaluation of SANS data simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Usage example:
      python fitness_evaluation.py sans_data.dat
            """
    )

    parser.add_argument(
        'dat_file',
        type=str,
        help='Source dat file to parse'
    )
    args = parser.parse_args()
    return args

def parse_sans_file(filepath):
    """
    Parse a SANS and extract q and I collumn

    Args:
        filepath (str): path to the .dat file

    Returns:
        tuple: (q, I) - 2 numpys arrays with q and I values
    """
    q_values = []
    I_values = []

    with open(filepath, 'r') as f:
        for line in f:
            # Ignorer les lignes de commentaire
            if line.startswith('#'):
                continue

            # Parser la ligne
            parts = line.split()
            if len(parts) >= 2:
                try:
                    q = float(parts[0])
                    I = float(parts[1])
                    q_values.append(q)
                    I_values.append(I)
                except ValueError:
                    # Ignorer les lignes qui ne peuvent pas être converties
                    continue

    # Convertir en arrays numpy
    q = np.array(q_values)
    I = np.array(I_values)

    return q, I


if __name__ == "__main__":
    args = parse_arguments()

    q2, I2 = parse_sans_file(args.dat_file)
    print(f"Number of points: {len(q2)}")
    print(f"first q: {q2[:5]}")
    print(f"first I: {I2[:5]}")
    print(f"q min: {q2.min()}, q max: {q2.max()}")
    print(f"I min: {I2.min()}, I max: {I2.max()}")

