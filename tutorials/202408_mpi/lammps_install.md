# Comparing Ranks vs. Threads using LAMMPS

>Lily de Melo

We need to set up our environment to use the mpich installed on the lab computers. We simpily need to export the following paths:
```bash
export PATH=/usr/local/mpich-4.1.1/bin:$PATH
export CPATH=/usr/local/mpich-4.1.1/include:$CPATH
```


Now create a file called `install_lammps.sh` and copy the following into it. You don’t need to know how it works for now. We just want to see the result of comparing threads and processes:

```Bash
#!/bin/bash

set -e  # Exit on any error

INSTALL_DIR=$HOME

if [ -d "$INSTALL_DIR/lammps" ]; then
    echo "lammps source repo already exists, skipping downloading"
else
    echo "Cloning lammps repository"
    cd "$INSTALL_DIR"
    git clone -b stable https://github.com/lammps/lammps.git
fi

cd "$INSTALL_DIR/lammps/src"

make clean-all
make yes-molecule yes-rigid yes-kspace yes-openmp

sed -i 's/-restrict/-Wrestrict/' MAKE/OPTIONS/Makefile.omp

make omp -j "$(nproc)"

cp lmp_omp "$INSTALL_DIR/lammps/bench/"
```

Run it:

```bash
bash lammps_install.sh
```

Now copy the contents of the following into a file called `run_lammps.sh`. Take a little time understand what is happening in this file; ask a mentor to clarify if unsure:

```bash
#!/bin/bash

INSTALL_DIR=$HOME

cd "$INSTALL_DIR/lammps/bench/"

#some commands needed in docker to run mpi programs as root
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

mpirun -np $1 ./lmp_omp -sf omp -pk omp $2 -in in.rhodo > lmp_serial_rhodo.out
cat lmp_serial_rhodo.out
```

- Here we can see that we can set the number of processes with `-np <number>` and the threads with `-pk omp <number>`.
- When we print out the output file we should see the runtime and number of threads and processors used.
  ```bash
  185.0% CPU use with 8 MPI tasks x 1 OpenMP threads
  .
  .
  Total wall time: 0:00:06
  ```
- You can play around until you get your fastest runtime - you can submit a screenshot on Moodle if you want to compare.

Run it:

```bash
bash lammps_run.sh <number of processors> <number of threads>

#example
bash lammps_run.sh 8 1
```
