#!/bin/env bash
# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
if [ $? -ne 0 ]; then
    trap 'echo "\"${last_command}\" command had an exit code $?."' EXIT
fi

source /home/intel/setvars.sh

cd /home/benchmarks/hpl/hpl-2.3
#check for existing make file
[ -f /home/benchmarks/hpl/hpl-2.3/Make.Linux_Intel64 ] && echo "make found" || echo "make not found"; cp ./setup/Make.Linux_Intel64 .

#some weird magic sed shit that changes the makefile
sed -i '70d' Make.Linux_Intel64
sed -i '70 i TOPdir = /home/benchmarks/hpl/hpl-2.3' Make.Linux_Intel64
sed -i 's#mkl/lib#lib#g' Make.Linux_Intel64
sed -i '95 i MKLROOT=/home/intel/mkl/latest' Make.Linux_Intel64
sed -i 's/-openmp/-qopenmp/g' Make.Linux_Intel64
sed -i '98d' Make.Linux_Intel64
sed -i '98 i LAinc = $(LAdir)/include' Make.Linux_Intel64

#making
make arch=Linux_Intel64 2>&1 | tee ./make.log

#check if executable exists
[ -f /home/benchmarks/hpl/hpl-2.3/bin/Linux_Intel64/xhpl ] && echo "hpl build successful" || echo "hpl build failed"

cd /home/benchmarks/hpl/hpl-2.3/bin/Linux_Intel64
cp /home/intel/mkl/latest/benchmarks/mp_linpack/HPL.dat .