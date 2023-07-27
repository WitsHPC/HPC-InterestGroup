#!/bin/env bash

# Check if OpenMPI is in the environment
if mpirun --version |& grep -q "Open MPI"; then
    mpi_info=$(mpirun --version | awk '/mpirun/ {print "OpenMPI - " $4}')
    echo "$mpi_info"
elif mpirun --version |& grep -q "MPICH"; then
    mpi_info=$(mpirun --version | awk '/mpirun/ {print "MPICH - " $4}')
    echo "$mpi_info"
else
    echo "No MPI found in environment"
fi
