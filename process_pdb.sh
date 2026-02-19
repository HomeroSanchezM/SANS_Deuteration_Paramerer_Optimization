#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <deuterated_pdbs_folder>"
    echo "Example: $0 my_analysis_deuterated_pdbs"
    exit 1
fi

# Get the input folder name
input_dir="$1"

# Check if the folder exists
if [ ! -d "$input_dir" ]; then
    echo "Error: The folder '$input_dir' does not exist!"
    exit 1
fi

# Extract only the folder name (without path)
folder_name=$(basename "$input_dir")

# Extract the prefix (xxx) from the folder name
# Remove the "_deuterated_pdbs" suffix if it exists, otherwise keep the full name
prefix="${folder_name%_deuterated_pdbs}"

# Create output directory in the CURRENT directory (where script is run)
output_dir="${prefix}_primus_out"

# If output directory exists, remove all files inside
if [ -d "$output_dir" ]; then
    echo "=== Output folder already exists ==="
    echo "Removing all files in: $output_dir"
    rm -rf "$output_dir"/*
    echo ""
fi

# Create output directories
mkdir -p "$output_dir"
mkdir -p "$output_dir/ref"

echo "=== Configuration ==="
echo "Input folder: $input_dir"
echo "Output folder: $output_dir"
echo "Prefix: $prefix"
echo ""

# Process files in the ref/ subfolder
if [ -d "$input_dir/ref" ]; then
    echo "=== Processing reference files ==="
    for pdb_file in "$input_dir/ref"/*.pdb; do
        # Check if .pdb files exist
        if [ ! -e "$pdb_file" ]; then
            echo "No .pdb files found in $input_dir/ref/"
            break
        fi
        
        # Extract the base name of the file
        basename=$(basename "$pdb_file" .pdb)
        
        # Build the output path
        output_file="$output_dir/ref/${basename}.dat"

        # Determine --d2o flag based on file name
        if [[ "$basename" == *"_total_deuteration" ]]; then
            d2o_flag="--d2o 1"
        elif [[ "$basename" == *"_total_protonation" ]]; then
            d2o_flag="--d2o 0"
        else
            d2o_flag=""
        fi
        
        # Execute Pepsi-SANS command
        echo "Processing: $pdb_file -> $output_file (d2o flag: '${d2o_flag:-none}')"
        ./Pepsi-SANS-Linux/Pepsi-SANS "$pdb_file" --hModel 3 --conc 5 $d2o_flag -o "$output_file"
    done
    echo ""
fi

# Process files at the root of the input folder
# Expected format: genXX_chrXXX_d2oXX_deutAAXX.pdb
echo "=== Processing main files ==="
for pdb_file in "$input_dir"/*.pdb; do
    # Check if .pdb files exist
    if [ ! -e "$pdb_file" ]; then
        echo "No .pdb files found in $input_dir/"
        break
    fi
    
    # Extract the base name of the file
    basename=$(basename "$pdb_file" .pdb)
    
    # Build the output path
    output_file="$output_dir/${basename}.dat"

    # Extract the d2o percentage from the filename (e.g., d2o24 -> 0.24)
    if [[ "$basename" =~ _d2o([0-9]+)_ ]]; then
        d2o_int="${BASH_REMATCH[1]}"
        # Convert integer percentage to decimal (e.g., 24 -> 0.24)
        d2o_flag="--d2o 0.$d2o_int"
    else
        echo "Warning: could not extract d2o value from filename '$basename', skipping --d2o flag"
        d2o_flag=""
    fi
    
    # Execute Pepsi-SANS command
    echo "Processing: $pdb_file -> $output_file (d2o flag: '${d2o_flag:-none}')"
    ../Pepsi-SANS-Linux/Pepsi-SANS "$pdb_file" --hModel 3 $d2o_flag -o "$output_file"
done

echo ""
echo "=== Simulation ended! ==="
echo "Results saved in: $output_dir"
