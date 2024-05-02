#!/bin/bash
# Clean the directory
make clean

# Remove the old files
rm *.bin
rm *.csv

# Compile the program
make

# Check if the compilation was successful
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed"
    exit 1
fi

echo ""

# Title for start of number generation
echo "-----------------------Starting Number Generation----------------------"

# Leave a blank line
echo ""

# Generate input data
./generator 20

# Calculate 2^n
n=$((2**20))

# Title for end of number generation
echo "Completed: Generated $n random numbers"

# Leave a blank line
echo ""
echo "------------------------------------------------------------------------"
echo ""
