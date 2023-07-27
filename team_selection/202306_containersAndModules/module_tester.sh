#!/bin/env bash

# Check if OpenMPI is in the environment
if command -v mpirun >/dev/null 2>&1; then
    # Get the MPI type and version
    mpi_info=$(mpirun --version | awk '/mpirun/ {if ($3 == "(Open MPI)") print "OpenMPI - " $4; else if ($3 == "(MPICH)") print "MPICH - " $4}')
    echo "$mpi_info"
else
    echo "No MPI found in environment"
fi
