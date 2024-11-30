#!/bin/bash
#SBATCH -n 4                    # Number of CPU cores
#SBATCH -N 1                    # Ensure all cores are on one machine
#SBATCH -D tmp                  # Working directory (change if necessary)
#SBATCH -t 4-00:05              # Runtime in D-HH:MM
#SBATCH -p tfg                  # Partition to submit to
#SBATCH --mem 24256             # Request 12 GB of RAM
#SBATCH -o %x_%u_%j.out         # STDOUT output file
#SBATCH -e %x_%u_%j.err         # STDERR error file
#SBATCH --gres gpu:3            # Request 1 GPU


# Run the Python script
python -m bitsandbytes



