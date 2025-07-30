#!/bin/env bash
source errorHandler
source spackEnv

ml hpl_2.3/intel-oneapi-compilers_2025.2.0/intel-oneapi-mpi_2021.16.0-mkl_2025.2.0
ml intel-oneapi-mpi_2021.16.0/gcc_12.3.0
ml intel-oneapi-mkl_2025.2.0/gcc_12.3.0

export OMP_NUM_THREADS=1

mpirun -n $(cores) xhpl
