#!/bin/env bash

# Check if wget is installed
if ! command -v wget &> /dev/null
then
    echo "wget not found, installing..."
    apt-get update
    apt-get install -y wget
else
    echo "wget is already installed"
fi

# Check if make is installed
if ! command -v make &> /dev/null
then
    echo "make not found, installing..."
    apt-get update
    apt-get install -y build-essential
else
    echo "make is already installed"
fi

# Check if ssh is installed
if ! command -v ssh &> /dev/null
then
    echo "ssh not found, installing..."
    apt-get update
    apt-get install -y ssh
else
    echo "ssh is already installed"
fi

# Check if tar is installed
if ! command -v tar &> /dev/null
then
    echo "tar not found, installing..."
    apt-get update
    apt-get install -y tar
else
    echo "tar is already installed"
fi



# Get the installation directory from the user
read -p "Enter the installation directory: " INSTALL_DIR
cd $INSTALL_DIR

# Download and extract OpenMPI
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz
tar -xzf openmpi-4.1.1.tar.gz

# Configure and install OpenMPI
cd openmpi-4.1.1
./configure --prefix=$INSTALL_DIR
make -j$(nproc)
make install

# Write the module file
mkdir -p $INSTALL_DIR/modules
cat << EOF > $INSTALL_DIR/modules/openmpi
#%Module1.0
prepend-path PATH $INSTALL_DIR/bin
prepend-path LD_LIBRARY_PATH $INSTALL_DIR/lib
EOF

# Append the moduleuse line to the bashrc
echo "module use --append $INSTALL_DIR/modules" >> ~/.bashrc
source ~/.bashrc
