# htop install script

## Setting up the environment

If you are not using a personal computer, you need to first set up an environment to use lmod. We will be using Docker.

Copy the following to a file called `docker_install.sh`:
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
Run the script:
```bash
bash ./docker_install.sh
```

Open and close the terminal and then run the following to see if docker has been installed correctly:
```bash
docker run hello-world
```

Now let's create an Unbuntu docker container using:
```bash
docker run -it ubuntu:latest bash
```

Then install all the necessary dependencies inside the docker container:
```bash
apt update && apt install -y lmod vim wget autotools-dev build-essential autoconf automake libncursesw5-dev
```
Export the file paths and test if lmod works:
```bash
echo "source /etc/profile.d/lmod.sh" >> ~/.bashrc
source ~/.bashrc

module avail
```
## Htop install script

Now that we've set up the environment, let's make a script called `htop_install.sh` and add the shebang first:

```bash
#!/bin/bash
```

Error checking:

```bash
# Exit immediately if a command exits with a non-zero status
set -e

# Handle exit and other signals
trap 'if [ $? -ne 0 ]; then echo "\"${last_command}\" command failed with exit code $?." 1>&2; fi' EXIT
trap 'echo "Script interrupted" 1>&2; exit 2' INT TERM
```

Check if Install directory exists:

```bash
INSTALL_DIR="/home/$USER/htop"

echo "Setting up the installation directory at $INSTALL_DIR"

# Create the installation directory if it doesn't exist
mkdir -p "$INSTALL_DIR"

cd "$INSTALL_DIR"
```

Download repo and unzip it:

```bash
if [ -f "$INSTALL_DIR/htop-3.3.0.tar.xz" ]; then
    echo "htop source repo already exists, skipping downloading"
else
    echo "Cloning htop repository"
    wget https://github.com/htop-dev/htop/releases/download/3.3.0/htop-3.3.0.tar.xz
fi

if [ -d "$INSTALL_DIR/htop-3.3.0" ]; then
    echo "htop directory already exists, skipping unziping"
else
    echo "Cloning htop repository"
    tar xvf htop-3.3.0.tar.xz
fi
```

Build htop:

```bash
echo "Building htop"
# Build htop
cd $INSTALL_DIR/htop-3.3.0

./autogen.sh
./configure --prefix=$INSTALL_DIR
make

echo "Installing htop"

# Install htop
make install

echo "htop has been installed successfully to $INSTALL_DIR"
```

Export the path:

```bash
mkdir -p $HOME/modulefiles

cat << EOF > $HOME/modulefiles/htop
#%Module1.0
prepend-path PATH $INSTALL_DIR/bin
EOF

echo "module use --append $HOME/modulefiles" >> ~/.bashrc
source ~/.bashrc

echo "Module file for htop has been created at ~/modulefiles/htop"
```

Or if lmod isn't working on your PC(test with ml avail) do the following:

```bash
echo "export PATH="$INSTALL_DIR/bin:$PATH"" >> ~/.bashrc
source ~/.bashrc
```
## Tutorial
Try to make a script to automate the installation of [LAMMPS](https://www.lammps.org/#gsc.tab=0)  
Build:
```
git clone -b stable https://github.com/lammps/lammps.git
```

```
cd lammps/src
```

```
make clean-all
make yes-molecule yes-rigid yes-kspace yes-openmp
```

```
sed -i 's/-restrict/-Wrestrict/' MAKE/OPTIONS/Makefile.omp
```

```
make omp -j "$(nproc)"
```

```
cp lmp_omp lammps/bench/
```

Run:
```
cd lammps/bench/
```

```
mpirun -np $1 ./lmp_omp -sf omp -pk omp $2 -in in.rhodo > lmp_serial_rhodo.out
cat lmp_serial_rhodo.out
```
