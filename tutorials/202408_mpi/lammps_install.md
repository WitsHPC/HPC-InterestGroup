# Comparing Ranks vs. Threads using LAMMPS

>Lily de Melo

We need to set up our environment the same way we did last week. Create a file called docker_install.sh and copy and paste the following into it. 

```bash
#!/bin/env bash
if ! command -v curl &> /dev/null
then
    echo "curl is not installed. Installing now..."
    temp_dir=$(mktemp -d)
    cd "$temp_dir"
    wget https://curl.se/download/curl-7.80.0.tar.gz
    tar xzf curl-7.80.0.tar.gz
    cd curl-7.80.0
    ./configure --prefix="$HOME/.local"
    make
    make install
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> ~/.bashrc
    source ~/.bashrc
else
    echo "curl is already installed."
fi
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Installing now..."
    curl -fsSL https://get.docker.com/rootless | sh
    echo "Docker installed"
    dir_to_add="/home/$USER/bin"
    echo "export PATH=\"\$PATH:$dir_to_add\"" >> ~/.bashrc
    source ~/.bashrc
else
    echo "Docker is already installed."
fi
```

Run it:

```bash
bash ./docker_install.sh
```

Test it:

```bash
docker run hello-world
```

Repeat this step from last week:

```Bash
docker run -it ubuntu:latest bash
```

We’ll need to install some additional dependencies:

```Bash
apt update && apt install -y lmod vim wget autotools-dev build-essential autoconf automake libncursesw5-dev cmake git openmpi-bin libopenmpi-dev libfftw3-dev libjpeg-dev libpng-dev libtiff-dev libx11-dev libxext-dev libxrender-dev
```

Create a file called `install_lammps.sh` and copy the following into it. You don’t need to know how it works for now. We just want to see the result of comparing threads and processes:

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

mpirun -np $1 lmp_omp -sf omp -pk omp $2 -in in.rhodo > lmp_serial_rhodo.out
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
