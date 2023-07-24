#!/bin/bash

# Get the installation directory from the user
read -p "Enter the installation directory: " INSTALL_DIR

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
