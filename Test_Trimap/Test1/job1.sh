#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p juliet
#SBATCH -t 50:30:00
#SBATCH -J test
#SBATCH -o test.o%j
srun -p juliet -N 1 -n 1 -c 1 mprof run conda run test_trimap.py
