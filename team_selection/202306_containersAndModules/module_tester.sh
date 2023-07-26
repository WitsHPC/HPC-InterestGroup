#!/bin/env bash

# Check if OpenMPI is in the environment
if command -v mpirun.openmpi >/dev/null 2>&1; then
    # Get the OpenMPI version
    openmpi_version=$(mpirun.openmpi --version | awk '/mpirun/ {print $NF}')
    echo "OpenMPI - $openmpi_version"
# Check if MPICH is in the environment
elif command -v mpirun.mpich >/dev/null 2>&1; then
    # Get the MPICH version
    mpich_version=$(mpirun.mpich --version | awk '/mpirun/ {print $NF}')
    echo "MPICH - $mpich_version"
else
    echo "No MPI found in environment"
fi
