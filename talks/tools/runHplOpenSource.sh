#!/bin/env bash
source errorHandler
source spackEnv

ml hpl_2.3/gcc_12.3.0/openmpi_5.0.8-openblas_0.3.30
ml openmpi_5.0.8/gcc_12.3.0
ml openblas_0.3.30/gcc_12.3.0

export OMP_NUM_THREADS=1

mpirun -n $(cores) xhpl
