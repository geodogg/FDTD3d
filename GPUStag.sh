#!/bin/bash
#SBATCH --account=def-spichard
#SBATCH --gres=gpu:1              # request GPU "generic resource", 4 on Cedar, 2 on Graham
#SBATCH --mem=16GB               # memory per node
#SBATCH --time=0-00:10            # time (DD-HH:MM)
./FDTD3d
