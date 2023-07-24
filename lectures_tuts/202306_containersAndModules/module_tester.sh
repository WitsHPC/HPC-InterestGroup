#!/bin/env bash

# Check if OpenMPI is in the environment
if command -v mpicc.openmpi >/dev/null 2>&1; then
    # Get the OpenMPI version
    openmpi_version=$(mpicc.openmpi --showme:version | awk '{print $NF}')
    echo "OpenMPI - $openmpi_version"
# Check if MPICH is in the environment
elif command -v mpicc.mpich >/dev/null 2>&1; then
    # Get the MPICH version
    mpich_version=$(mpicc.mpich -v 2>&1 | awk '/version/ {print $NF}')
    echo "MPICH - $mpich_version"
else
    echo "No MPI found in environment"
fi
